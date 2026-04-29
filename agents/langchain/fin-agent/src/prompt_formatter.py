"""
Prompt formatter for Llama 3 chat template.

This module builds prompts according to Llama 3's official chat template format,
including proper special tokens and message structure.
"""

import logging
from typing import List, Any, Dict, Optional
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from .tool_schema_formatter import build_tool_instruction

logger = logging.getLogger(__name__)


def build_llama3_prompt(
    messages: List[BaseMessage], 
    tool_instruction: str = "",
    tools: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Build a Llama 3 formatted prompt from LangChain messages.
    
    Formats messages according to Llama 3's chat template with proper special tokens:
    - Starts with <|begin_of_text|>
    - Wraps each message with <|start_header_id|>role<|end_header_id|>content<|eot_id|>
    - Ends with <|start_header_id|>assistant<|end_header_id|> to prompt response
    
    Args:
        messages: List of LangChain message objects (HumanMessage, AIMessage, 
                  SystemMessage, ToolMessage)
        tool_instruction: Optional pre-formatted tool instruction text to inject into system message.
                         If both tool_instruction and tools are provided, tool_instruction takes precedence.
        tools: Optional list of LangChain tool definitions. If provided, tool schemas will be
               formatted using tool_schema_formatter and injected into the system message.
        
    Returns:
        Formatted prompt string with Llama 3 special tokens
        
    Example:
        >>> messages = [
        ...     SystemMessage(content="You are a helpful assistant"),
        ...     HumanMessage(content="Hello!"),
        ... ]
        >>> prompt = build_llama3_prompt(messages)
        >>> prompt.startswith("<|begin_of_text|>")
        True
        >>> "<|start_header_id|>system<|end_header_id|>" in prompt
        True
        
        >>> # With tools
        >>> tools = [{
        ...     "name": "get_weather",
        ...     "description": "Get weather info",
        ...     "parameters": {"type": "object", "properties": {}}
        ... }]
        >>> prompt = build_llama3_prompt(messages, tools=tools)
        >>> "You have access to the following tools:" in prompt
        True
    """
    # Check if tools are provided and format them
    if not tool_instruction and tools:
        try:
            tool_instruction = build_tool_instruction(tools)
        except Exception as e:
            # Graceful degradation: log warning and continue without tool schemas
            # This ensures the agent can still respond even if tool formatting fails
            logger.warning(
                f"Tool schema formatting failed, falling back to basic instruction following. "
                f"Error: {type(e).__name__}: {str(e)}. "
                f"Tool count: {len(tools) if tools else 0}. "
                f"Diagnostic info: {{'error_type': '{type(e).__name__}', 'error_message': '{str(e)}', 'tools_provided': {len(tools) if tools else 0}}}"
            )
            tool_instruction = ""  # Fall back to no tool instruction
    
    # Start with begin_of_text token
    prompt_parts = ["<|begin_of_text|>"]
    
    # Track if we've seen a system message
    has_system_message = False
    
    # Process each message
    for message in messages:
        # Determine the role
        if isinstance(message, HumanMessage):
            role = "user"
            content = message.content
        elif isinstance(message, AIMessage):
            role = "assistant"
            content = message.content
        elif isinstance(message, SystemMessage):
            role = "system"
            content = message.content
            has_system_message = True
            
            # Inject tool instruction into system message if provided
            if tool_instruction:
                content = f"{content}\n\n{tool_instruction}"
        elif isinstance(message, ToolMessage):
            # Tool messages are formatted as user messages with tool result
            role = "user"
            # Include tool name if available
            tool_name = getattr(message, 'name', None) or 'tool'
            content = f"Tool result from {tool_name}:\n{message.content}"
        else:
            # Fallback for unknown message types
            role = "user"
            content = str(message.content)
        
        # Format the message with Llama 3 special tokens
        message_text = (
            f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            f"{content}<|eot_id|>"
        )
        prompt_parts.append(message_text)
    
    # If no system message and we have tool instructions, add a system message
    if not has_system_message and tool_instruction:
        system_message = (
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{tool_instruction}<|eot_id|>"
        )
        # Insert after begin_of_text
        prompt_parts.insert(1, system_message)
    
    # Append assistant header to prompt the model's response
    prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    
    return "".join(prompt_parts)
