"""
Content handler for Llama 3 function calling with DJL/vLLM native support.

This module provides a content handler that works with SageMaker's DJL container
with vLLM backend, which has native tool calling support. The handler transforms
LangChain messages to OpenAI-compatible format and parses the native tool call responses.
"""

import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage
from langchain_aws.chat_models.sagemaker_endpoint import ChatModelContentHandler

logger = logging.getLogger(__name__)


class LlamaFunctionCallingHandler(ChatModelContentHandler):
    """
    Content handler for DJL/vLLM native tool calling support.
    
    This handler transforms LangChain messages into OpenAI-compatible format
    that the DJL container with vLLM backend expects, and parses the native
    tool call responses back into LangChain AIMessage format.
    
    The DJL container with OPTION_ENABLE_AUTO_TOOL_CHOICE=true handles:
    - Chat template formatting
    - Tool schema injection
    - Tool call parsing
    
    This handler just needs to:
    - Transform LangChain messages to OpenAI format
    - Extract tools from model_kwargs
    - Parse the OpenAI-compatible response
    
    Attributes:
        content_type: MIME type for request payload ("application/json")
        accepts: MIME type for response payload ("application/json")
    """
    
    content_type = "application/json"
    accepts = "application/json"
    
    def transform_input(
        self, 
        messages: List[BaseMessage], 
        model_kwargs: Dict[str, Any]
    ) -> bytes:
        """
        Transform LangChain messages to DJL/vLLM OpenAI-compatible format.
        
        The DJL container expects:
        {
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "tool_name",
                        "description": "...",
                        "parameters": {...}
                    }
                }
            ],
            "max_tokens": 512,
            "temperature": 0.1,
            ...
        }
        
        Args:
            messages: List of LangChain message objects or dicts
            model_kwargs: Additional parameters including tools
            
        Returns:
            JSON bytes for DJL container
            
        Example:
            >>> handler = LlamaFunctionCallingHandler()
            >>> messages = [HumanMessage(content="Hello")]
            >>> payload = handler.transform_input(messages, {})
            >>> isinstance(payload, bytes)
            True
        """
        
        # Convert LangChain messages to OpenAI format
        openai_messages = []
        for msg in messages:
            # Handle both Message objects and dicts
            if isinstance(msg, dict):
                # Already a dict, use as-is
                openai_messages.append(msg)
                continue
            
            role = self._get_message_role(msg)
            content = str(msg.content) if msg.content else ""
            
            message_dict = {
                "role": role,
                "content": content
            }
            
            # Handle tool calls in assistant messages
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                message_dict["tool_calls"] = [
                    {
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"])
                        }
                    }
                    for i, tc in enumerate(msg.tool_calls)
                ]
            
            # Handle tool messages
            if hasattr(msg, 'tool_call_id'):
                message_dict["tool_call_id"] = msg.tool_call_id
                message_dict["role"] = "tool"
            
            openai_messages.append(message_dict)
        
        # Build request payload
        payload = {
            "messages": openai_messages
        }
        
        # Extract tools from model_kwargs if present
        tools = model_kwargs.get("tools", [])
        if tools:
            # Convert LangChain tools to OpenAI format
            openai_tools = []
            for tool in tools:
                # Handle both dict and LangChain tool objects
                if hasattr(tool, 'name'):
                    # LangChain tool object or MCP tool
                    parameters = {}
                    if hasattr(tool, 'args_schema'):
                        args_schema = tool.args_schema
                        # Check if args_schema is already a dict (MCP tools) or needs .schema() call (LangChain tools)
                        if isinstance(args_schema, dict):
                            parameters = args_schema
                        elif hasattr(args_schema, 'schema'):
                            parameters = args_schema.schema()
                        else:
                            logger.warning(f"Tool {tool.name} has args_schema but it's neither dict nor has .schema() method")
                    
                    tool_dict = {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": parameters
                        }
                    }
                else:
                    # Already a dict
                    tool_dict = {
                        "type": "function",
                        "function": {
                            "name": tool.get("name"),
                            "description": tool.get("description"),
                            "parameters": tool.get("parameters", {})
                        }
                    }
                openai_tools.append(tool_dict)
            
            payload["tools"] = openai_tools
        
        # Add generation parameters
        payload["max_tokens"] = model_kwargs.get("max_tokens", model_kwargs.get("max_new_tokens", 512))
        payload["temperature"] = model_kwargs.get("temperature", 0.1)
        payload["top_p"] = model_kwargs.get("top_p", 0.9)
        
        # Enable parallel tool calling if tools are present
        if tools:
            payload["parallel_tool_calls"] = True
        
        # CRITICAL DEBUG: Log payload details to diagnose context length issue
        payload_json = json.dumps(payload)
        payload_bytes = payload_json.encode("utf-8")
        
        # Calculate approximate token counts
        total_chars = len(payload_json)
        approx_tokens = total_chars // 4  # Rough estimate: 1 token ≈ 4 chars
        
        # Count messages and their sizes
        message_count = len(openai_messages)
        message_sizes = []
        for i, msg in enumerate(openai_messages):
            msg_str = json.dumps(msg)
            msg_chars = len(msg_str)
            msg_tokens = msg_chars // 4
            message_sizes.append((i, msg["role"], msg_chars, msg_tokens))
        
        logger.info("="*80)
        logger.info("🔍 TRANSFORM_INPUT DEBUG - PAYLOAD ANALYSIS")
        logger.info("="*80)
        logger.info(f"📊 Message count: {message_count}")
        logger.info(f"📊 Total payload size: {total_chars} chars (~{approx_tokens} tokens)")
        logger.info(f"📊 Max tokens requested: {payload['max_tokens']}")
        logger.info(f"📊 Estimated total: ~{approx_tokens + payload['max_tokens']} tokens")
        logger.info("\n📋 Message breakdown:")
        for idx, role, chars, tokens in message_sizes:
            logger.info(f"  Message {idx} ({role}): {chars} chars (~{tokens} tokens)")
        
        if tools:
            tools_json = json.dumps(payload.get("tools", []))
            tools_chars = len(tools_json)
            tools_tokens = tools_chars // 4
            logger.info(f"\n🔧 Tools payload: {tools_chars} chars (~{tools_tokens} tokens)")
        
        logger.info("="*80)
        
        logger.debug(f"DJL request payload: {json.dumps(payload, indent=2)}")
        
        return payload_bytes
    
    def transform_output(self, output: bytes) -> AIMessage:
        """
        Transform DJL/vLLM response to LangChain AIMessage.
        
        The DJL container returns OpenAI-compatible format:
        {
            "id": "chatcmpl-...",
            "object": "chat.completion",
            "created": 1234567890,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "...",
                        "tool_calls": [
                            {
                                "id": "call_...",
                                "type": "function",
                                "function": {
                                    "name": "tool_name",
                                    "arguments": "{...}"
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ]
        }
        
        Args:
            output: JSON bytes or StreamingBody from DJL container
            
        Returns:
            LangChain AIMessage with tool_calls or content
            
        Raises:
            ValueError: If response format is invalid or malformed
            
        Example:
            >>> handler = LlamaFunctionCallingHandler()
            >>> response = b'{"choices": [{"message": {"role": "assistant", "content": "Hello!"}}]}'
            >>> message = handler.transform_output(response)
            >>> message.content
            'Hello!'
        """
        try:
            # Handle StreamingBody from boto3
            if hasattr(output, 'read'):
                output_bytes = output.read()
            else:
                output_bytes = output
            
            response = json.loads(output_bytes.decode("utf-8"))
            
            # DEBUG: Print full response structure
            print(f"\n[CONTENT_HANDLER DEBUG] Full response:")
            print(json.dumps(response, indent=2)[:1000])
            
            logger.debug(f"DJL response: {json.dumps(response, indent=2)}")
            
            # Extract the message from the first choice
            if "choices" not in response or len(response["choices"]) == 0:
                error_details = {
                    "error_type": "InvalidResponseFormat",
                    "expected": "response with 'choices' array",
                    "received": list(response.keys()) if isinstance(response, dict) else type(response).__name__,
                    "raw_output_preview": str(output_bytes[:200])
                }
                logger.error(f"Invalid response format: {error_details}")
                raise ValueError("No choices in response")
            
            choice = response["choices"][0]
            message = choice.get("message", {})
            
            # DEBUG: Print message structure
            print(f"\n[CONTENT_HANDLER DEBUG] Message structure:")
            print(f"  - role: {message.get('role')}")
            print(f"  - content: {str(message.get('content', ''))[:200]}")
            print(f"  - tool_calls present: {'tool_calls' in message}")
            if 'tool_calls' in message:
                print(f"  - tool_calls: {message['tool_calls']}")
            
            # Extract content
            content = message.get("content", "")
            
            # Extract tool calls if present
            tool_calls = []
            if "tool_calls" in message and message["tool_calls"]:
                for tc in message["tool_calls"]:
                    function = tc.get("function", {})
                    try:
                        raw_args_str = function.get("arguments", "{}")
                        args = json.loads(raw_args_str)
                        
                        # DEBUG: Log before type coercion
                        print(f"\n[TYPE_COERCION DEBUG] Before fix:")
                        print(f"  Raw arguments string: {raw_args_str}")
                        print(f"  Parsed args: {args}")
                        print(f"  Arg types: {[(k, type(v).__name__) for k, v in args.items()]}")
                        
                        # Fix: Convert string-formatted types to actual types
                        # The model sometimes outputs:
                        # - "['item1', 'item2']" instead of ["item1", "item2"]
                        # - "true"/"false" instead of true/false
                        # - "3" instead of 3
                        args = self._fix_string_arrays(args)
                        
                        # DEBUG: Log after type coercion
                        print(f"\n[TYPE_COERCION DEBUG] After fix:")
                        print(f"  Fixed args: {args}")
                        print(f"  Arg types: {[(k, type(v).__name__) for k, v in args.items()]}")
                        
                        tool_calls.append({
                            "name": function.get("name"),
                            "args": args,
                            "id": tc.get("id", "")
                        })
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse tool call arguments: {e}")
                        logger.debug(f"Raw arguments: {function.get('arguments')}")
                        # Skip malformed tool calls
                        continue
            
            # FALLBACK: Parse tool calls from content if not in tool_calls field
            # This handles cases where the model generates <tool_call> tags in content
            if not tool_calls and content:
                logger.info("No tool_calls in response, attempting to parse from content field")
                tool_calls = self._parse_tool_calls_from_content(content)
            
            # Create AIMessage
            if tool_calls:
                logger.info(f"Detected {len(tool_calls)} tool call(s): {[tc['name'] for tc in tool_calls]}")
                return AIMessage(content=content or "", tool_calls=tool_calls)
            else:
                logger.debug("No valid tool calls detected, returning as text response")
                return AIMessage(content=content)
        
        except json.JSONDecodeError as e:
            # Handle StreamingBody for error reporting
            if hasattr(output, 'read'):
                output_bytes_for_error = output.read()
                output_preview = str(output_bytes_for_error[:500])
                output_length = len(output_bytes_for_error)
            else:
                output_preview = str(output[:500])
                output_length = len(output)
            
            error_details = {
                "error_type": "JSONDecodeError",
                "error_message": str(e),
                "error_location": f"line {e.lineno}, column {e.colno}" if hasattr(e, 'lineno') else "unknown",
                "raw_output_preview": output_preview,
                "output_length": output_length,
            }
            logger.error(f"Failed to parse DJL response: {error_details}")
            raise ValueError(f"Invalid JSON response from DJL: {e}") from e
        
        except KeyError as e:
            error_details = {
                "error_type": "KeyError",
                "missing_key": str(e),
                "response_structure": str(response)[:200] if 'response' in locals() else "N/A"
            }
            logger.error(f"Missing expected field in response: {error_details}")
            logger.debug(f"Response: {response if 'response' in locals() else 'N/A'}")
            raise ValueError(f"Malformed response structure: {e}") from e
        
        except Exception as e:
            # Handle StreamingBody for error reporting
            if hasattr(output, 'read'):
                output_preview = str(output.read()[:200])
            else:
                output_preview = str(output[:200])
            
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "output_preview": output_preview,
            }
            logger.error(f"Unexpected error in transform_output: {error_details}")
            logger.exception("Full exception traceback:")
            raise
    
    def _parse_tool_calls_from_content(self, content: str) -> List[Dict[str, Any]]:
        """
        Fallback parser for tool calls embedded in content field.
        
        Some models (like Llama 3.1 8B) generate tool calls in various formats:
        1. XML-wrapped: <tool_call>{"name": "...", "parameters": {...}}</tool_call>
        2. Plain JSON: {"name": "...", "parameters": {...}}
        3. Multiple on separate lines
        
        This method extracts and parses these tool calls.
        
        Args:
            content: The content string that may contain tool calls
            
        Returns:
            List of tool call dictionaries in LangChain format
        """
        import re
        
        tool_calls = []
        
        # PATTERN 1: XML-wrapped tool calls <tool_call>...</tool_call>
        xml_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        xml_matches = re.findall(xml_pattern, content, re.DOTALL)
        
        if xml_matches:
            logger.info(f"Found {len(xml_matches)} XML-wrapped tool call(s) in content field")
            
            for i, match in enumerate(xml_matches):
                try:
                    # Parse the JSON inside the tool_call tags
                    tool_data = json.loads(match)
                    
                    # Extract name and parameters/arguments
                    name = tool_data.get("name")
                    # Handle both "parameters" and "arguments" keys
                    args = tool_data.get("parameters") or tool_data.get("arguments") or {}
                    
                    if not name:
                        logger.warning(f"Tool call {i+1} missing 'name' field, skipping")
                        continue
                    
                    # Generate unique call ID
                    import uuid
                    call_id = f"call_{uuid.uuid4().hex[:16]}"
                    
                    tool_calls.append({
                        "name": name,
                        "args": args,
                        "id": call_id
                    })
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse XML-wrapped tool call {i+1}: {e}")
                    continue
        
        # PATTERN 2: Plain JSON tool calls (no XML wrapper)
        # Look for {"name": "...", "parameters": {...}} or {"name": "...", "arguments": {...}}
        # This handles cases where model outputs plain JSON without XML tags
        # Use a more robust approach: find all lines that look like complete JSON objects
        if not tool_calls:  # Only try this if XML parsing didn't find anything
            lines = content.split('\n')
            for line_num, line in enumerate(lines):
                line = line.strip()
                # Check if line looks like a JSON object with "name" field
                if line.startswith('{') and '"name"' in line and ('"parameters"' in line or '"arguments"' in line):
                    try:
                        # Try to parse the entire line as JSON
                        tool_data = json.loads(line)
                        
                        # Extract name and parameters/arguments
                        name = tool_data.get("name")
                        args = tool_data.get("parameters") or tool_data.get("arguments") or {}
                        
                        if not name:
                            logger.warning(f"Plain JSON tool call on line {line_num+1} missing 'name' field, skipping")
                            continue
                        
                        # Apply type coercion to arguments
                        args = self._fix_string_arrays(args)
                        
                        # Generate unique call ID
                        import uuid
                        call_id = f"call_{uuid.uuid4().hex[:16]}"
                        
                        tool_calls.append({
                            "name": name,
                            "args": args,
                            "id": call_id
                        })
                        
                        logger.info(f"Parsed plain JSON tool call from line {line_num+1}: {name}({args})")
                        
                    except json.JSONDecodeError as e:
                        logger.debug(f"Failed to parse line {line_num+1} as JSON tool call: {e}")
                        continue
        
        if tool_calls:
            logger.info(f"Successfully parsed {len(tool_calls)} tool call(s) from content field")
        else:
            logger.debug("No tool calls found in content field")
        
        return tool_calls
    
    def _fix_string_arrays(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix string-formatted arrays, booleans, and numbers in tool arguments.
        
        The model sometimes outputs Python string representations instead of JSON types:
        - "['item1', 'item2']" instead of ["item1", "item2"]
        - "True" or "False" instead of true or false
        - "4" or "10" instead of 4 or 10
        
        This method detects and converts these string representations to proper types.
        
        Args:
            args: Tool arguments dictionary
            
        Returns:
            Fixed arguments dictionary with proper types
            
        Example:
            >>> handler = LlamaFunctionCallingHandler()
            >>> args = {"param": "['profile', 'financials']", "flag": "True", "count": "4"}
            >>> fixed = handler._fix_string_arrays(args)
            >>> fixed
            {'param': ['profile', 'financials'], 'flag': True, 'count': 4}
        """
        import ast
        
        fixed_args = {}
        for key, value in args.items():
            if isinstance(value, str):
                # Check for boolean strings (both Python and JSON formats)
                if value in ("True", "False", "true", "false"):
                    converted = value in ("True", "true")
                    logger.info(f"Converted string boolean '{value}' to actual boolean: {converted}")
                    fixed_args[key] = converted
                    continue
                
                # Check if it looks like a Python list representation
                if value.startswith('[') and value.endswith(']'):
                    try:
                        # Try to parse as Python literal (handles ['item1', 'item2'])
                        parsed = ast.literal_eval(value)
                        if isinstance(parsed, list):
                            logger.info(f"Converted string array '{value}' to actual array: {parsed}")
                            fixed_args[key] = parsed
                            continue
                    except (ValueError, SyntaxError):
                        # Not a valid Python literal, keep as string
                        pass
                
                # Check if it looks like a number string
                if value.isdigit():
                    # Integer
                    converted = int(value)
                    logger.info(f"Converted string integer '{value}' to actual integer: {converted}")
                    fixed_args[key] = converted
                    continue
                
                # Check for float strings
                try:
                    if '.' in value:
                        converted = float(value)
                        logger.info(f"Converted string float '{value}' to actual float: {converted}")
                        fixed_args[key] = converted
                        continue
                except ValueError:
                    # Not a valid float, keep as string
                    pass
            
            # Keep original value if not converted
            fixed_args[key] = value
        
        return fixed_args
    
    def _get_message_role(self, message: BaseMessage) -> str:
        """
        Get OpenAI role from LangChain message type.
        
        Args:
            message: LangChain message object
            
        Returns:
            OpenAI role string: "user", "assistant", "system", or "tool"
        """
        from langchain_core.messages import (
            HumanMessage,
            AIMessage,
            SystemMessage,
            ToolMessage,
        )
        
        if isinstance(message, HumanMessage):
            return "user"
        elif isinstance(message, AIMessage):
            return "assistant"
        elif isinstance(message, SystemMessage):
            return "system"
        elif isinstance(message, ToolMessage):
            return "tool"
        else:
            # Default to user for unknown types
            logger.warning(f"Unknown message type: {type(message).__name__}, defaulting to 'user' role")
            return "user"
