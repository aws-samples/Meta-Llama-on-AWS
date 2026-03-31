"""
Response parser for extracting tool calls from Llama 3 model outputs.

This module provides functionality to parse tool call JSON from model-generated
text and transform it into LangChain-compatible tool call structures.
"""

import json
import re
import uuid
from typing import List, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def parse_tool_call(text: str) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """
    Parse tool calls from model output text.
    
    Searches for JSON patterns matching the tool call format:
    {"tool": "tool_name", "args": {...}}
    
    Args:
        text: Raw text output from the model
        
    Returns:
        Tuple of (tool_calls, remaining_text):
        - tool_calls: List of dicts with 'name', 'args', 'id' keys, or None if no valid calls
        - remaining_text: Text with tool call JSON removed, or original text if no calls found
        
    Example:
        >>> text = '{"tool": "yahoo_stock_price", "args": {"ticker": "AAPL"}}'
        >>> calls, remaining = parse_tool_call(text)
        >>> calls[0]['name']
        'yahoo_stock_price'
        >>> calls[0]['args']
        {'ticker': 'AAPL'}
    """
    tool_calls = []
    errors = []
    positions_to_remove = []
    
    # Find all potential tool call patterns
    # Look for {"tool": "...", "args": ...}
    pattern = r'\{"tool":\s*"([^"]+)",\s*"args":\s*'
    
    for match in re.finditer(pattern, text):
        tool_name = match.group(1)
        start_pos = match.start()
        args_start = match.end()
        
        # Find the matching closing brace for the args object
        try:
            # Use brace counting to find the complete args object
            brace_count = 0
            i = args_start
            in_string = False
            escape_next = False
            args_end = -1
            
            while i < len(text):
                char = text[i]
                
                if escape_next:
                    escape_next = False
                    i += 1
                    continue
                
                if char == '\\':
                    escape_next = True
                elif char == '"':
                    in_string = not in_string
                elif not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        if brace_count == 0:
                            # This is the closing brace for the args object
                            args_end = i
                            break
                        else:
                            brace_count -= 1
                
                i += 1
            
            if args_end > args_start:
                # Extract the args JSON
                args_json = text[args_start:args_end]
                
                # Now find the closing brace for the entire tool call object
                i = args_end
                # Skip whitespace
                while i < len(text) and text[i].isspace():
                    i += 1
                
                if i < len(text) and text[i] == '}':
                    end_pos = i + 1
                    
                    # Try to parse the args
                    try:
                        args = json.loads(args_json)
                        
                        # Generate unique call ID
                        call_id = f"call_{uuid.uuid4().hex[:16]}"
                        
                        tool_calls.append({
                            "name": tool_name,
                            "args": args,
                            "id": call_id
                        })
                        
                        positions_to_remove.append((start_pos, end_pos))
                        
                    except json.JSONDecodeError as e:
                        error_msg = f"Invalid JSON in tool call for '{tool_name}': {args_json}"
                        logger.warning(error_msg)
                        errors.append({
                            "tool": tool_name,
                            "error_type": "JSONDecodeError",
                            "error_message": str(e),
                            "error_location": f"line {e.lineno}, column {e.colno}" if hasattr(e, 'lineno') else "unknown",
                            "raw_args": args_json,
                            "position_in_text": f"characters {start_pos}-{end_pos}"
                        })
        
        except Exception as e:
            error_msg = f"Error parsing tool call for '{tool_name}': {type(e).__name__}: {e}"
            logger.warning(error_msg)
            errors.append({
                "tool": tool_name,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "position_in_text": f"starting at character {start_pos}"
            })
            continue
    
    # Log parsing errors if any occurred
    if errors:
        logger.info(f"Tool call parsing encountered {len(errors)} error(s):")
        for error in errors:
            logger.info(f"  - Tool: {error['tool']}")
            logger.info(f"    Error Type: {error['error_type']}")
            logger.info(f"    Error Message: {error['error_message']}")
            if 'error_location' in error:
                logger.info(f"    Error Location: {error['error_location']}")
            if 'position_in_text' in error:
                logger.info(f"    Position in Text: {error['position_in_text']}")
            if 'raw_args' in error:
                logger.info(f"    Raw Arguments: {error['raw_args']}")
    
    # If we found valid tool calls, remove them from the text
    if tool_calls:
        # Remove positions in reverse order to maintain indices
        remaining_text = text
        for start, end in reversed(positions_to_remove):
            remaining_text = remaining_text[:start] + remaining_text[end:]
        remaining_text = remaining_text.strip()
        return tool_calls, remaining_text
    
    # No valid tool calls found
    return None, text
