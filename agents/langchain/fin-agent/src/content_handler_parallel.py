"""
Modified Content Handler with Parallel Tool Calling Support

This version adds parallel_tool_calls=True to the request payload to enable
multiple tool calls per response from the LMI container.

Key Changes:
1. Added parallel_tool_calls parameter to transform_input
2. Removed single-tool validation from transform_output
3. Handles multiple tool calls in a single response

Usage:
    Replace src/content_handler.py with this file to enable parallel tool calling.
"""

import json
from typing import Any, Dict, List
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import AIMessage


class SageMakerLLMContentHandler:
    """
    Content handler for SageMaker LLM endpoint with parallel tool calling support.
    
    This handler transforms LangChain messages into the format expected by the
    LMI container and parses the responses back into LangChain format.
    """
    
    content_type = "application/json"
    accepts = "application/json"
    
    def transform_input(self, prompt: str, model_kwargs: Dict[str, Any]) -> bytes:
        """
        Transform LangChain input into SageMaker endpoint format.
        
        Args:
            prompt: The input prompt (unused, we use messages from model_kwargs)
            model_kwargs: Dictionary containing messages, tools, and other parameters
            
        Returns:
            JSON-encoded bytes for the SageMaker endpoint
        """
        # Extract parameters
        messages = model_kwargs.get("messages", [])
        tools = model_kwargs.get("tools", [])
        
        # Build request payload
        payload = {
            "messages": messages,
            "max_tokens": model_kwargs.get("max_tokens", 2048),
            "temperature": model_kwargs.get("temperature", 0.7),
            "top_p": model_kwargs.get("top_p", 0.9),
        }
        
        # Add tools if provided
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = model_kwargs.get("tool_choice", "auto")
            
            # ← KEY CHANGE: Enable parallel tool calling
            payload["parallel_tool_calls"] = True
        
        return json.dumps(payload).encode("utf-8")
    
    def transform_output(self, output: bytes) -> Dict[str, Any]:
        """
        Transform SageMaker endpoint output into LangChain format.
        
        Args:
            output: Raw bytes from the SageMaker endpoint
            
        Returns:
            Dictionary with parsed response data
        """
        # Parse JSON response
        response = json.loads(output.decode("utf-8"))
        
        # Extract the message from the response
        if "choices" in response and len(response["choices"]) > 0:
            message = response["choices"][0]["message"]
            
            # Check if there are tool calls
            if "tool_calls" in message and message["tool_calls"]:
                tool_calls = message["tool_calls"]
                
                # ← KEY CHANGE: Handle multiple tool calls
                # Return all tool calls, not just the first one
                return {
                    "role": "assistant",
                    "content": message.get("content", ""),
                    "tool_calls": tool_calls,  # Return all tool calls
                    "raw_response": response
                }
            else:
                # No tool calls, return content
                return {
                    "role": "assistant",
                    "content": message.get("content", ""),
                    "raw_response": response
                }
        
        # Fallback if response format is unexpected
        return {
            "role": "assistant",
            "content": str(response),
            "raw_response": response
        }


def parse_tool_calls(response: Dict[str, Any]) -> List[AgentAction]:
    """
    Parse tool calls from the response into LangChain AgentAction objects.
    
    Args:
        response: Parsed response from transform_output
        
    Returns:
        List of AgentAction objects, one for each tool call
    """
    tool_calls = response.get("tool_calls", [])
    
    if not tool_calls:
        return []
    
    actions = []
    
    # ← KEY CHANGE: Process all tool calls, not just the first one
    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        tool_name = function.get("name", "")
        
        # Parse arguments
        try:
            tool_input = json.loads(function.get("arguments", "{}"))
        except json.JSONDecodeError:
            tool_input = {"raw_arguments": function.get("arguments", "")}
        
        # Create AgentAction
        action = AgentAction(
            tool=tool_name,
            tool_input=tool_input,
            log=f"Calling tool: {tool_name} with input: {tool_input}",
        )
        
        actions.append(action)
    
    return actions


def create_agent_finish(response: Dict[str, Any]) -> AgentFinish:
    """
    Create an AgentFinish object from the response.
    
    Args:
        response: Parsed response from transform_output
        
    Returns:
        AgentFinish object with the final response
    """
    content = response.get("content", "")
    
    return AgentFinish(
        return_values={"output": content},
        log=content,
    )


# Example usage
if __name__ == "__main__":
    # Test the content handler
    handler = SageMakerLLMContentHandler()
    
    # Test input transformation
    model_kwargs = {
        "messages": [
            {"role": "user", "content": "What are the stock prices for AAPL and MSFT?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "yahoo_stock_price",
                    "description": "Get stock price",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"}
                        },
                        "required": ["ticker"]
                    }
                }
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.7,
    }
    
    input_bytes = handler.transform_input("", model_kwargs)
    print("Input payload:")
    print(json.dumps(json.loads(input_bytes), indent=2))
    
    # Test output transformation with multiple tool calls
    output_bytes = json.dumps({
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "yahoo_stock_price",
                                "arguments": '{"ticker": "AAPL"}'
                            }
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "yahoo_stock_price",
                                "arguments": '{"ticker": "MSFT"}'
                            }
                        }
                    ]
                }
            }
        ]
    }).encode("utf-8")
    
    parsed = handler.transform_output(output_bytes)
    print("\nParsed output:")
    print(json.dumps(parsed, indent=2, default=str))
    
    # Test parsing tool calls
    actions = parse_tool_calls(parsed)
    print(f"\nParsed {len(actions)} tool calls:")
    for i, action in enumerate(actions, 1):
        print(f"  {i}. {action.tool}: {action.tool_input}")
