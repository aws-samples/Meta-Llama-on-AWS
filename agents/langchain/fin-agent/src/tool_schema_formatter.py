"""
Tool schema formatter for Llama 3 function calling.

This module converts LangChain tool definitions into a human-readable format
that Llama 3 can understand and follow for function calling.
"""

from typing import Any, Dict, List


# Template for tool instructions that will be injected into the system prompt
TOOL_INSTRUCTION_TEMPLATE = """You have access to the following tools:

{tool_schemas}

IMPORTANT: You can call tools multiple times in sequence to gather all needed information.

Examples of multi-tool usage:
- To compare two stocks: call get_stock_price twice with different tickers
- To get comprehensive data: first call get_ticker, then call get_price with that ticker
- To analyze multiple companies: call tools for each company separately

To call a tool, respond with a JSON object in this EXACT format:
{{"tool": "tool_name", "args": {{"param1": "value1", "param2": "value2"}}}}

After receiving tool results, you should:
- Call ANOTHER tool if you need more information to fully answer the question
- Provide your final answer ONLY when you have gathered all necessary information

You may call as many tools as needed to completely answer the user's question."""


def format_tool_schema(tool: Dict[str, Any]) -> str:
    """
    Format a single tool schema into human-readable text for Llama 3.
    
    Extracts tool name, description, and parameters from a LangChain tool definition
    and formats them into a structured text format that the model can reliably parse.
    
    Args:
        tool: LangChain tool definition dictionary with keys:
            - name: Tool name (string)
            - description: Tool description (string)
            - parameters: JSON schema object with properties and required fields
    
    Returns:
        Formatted tool schema string in the format:
        - tool_name(param1: type, param2: type): Description
          Parameters:
            * param1 (required): Parameter description
            * param2 (optional): Parameter description
    
    Example:
        Input:
        {
            "name": "yahoo_stock_price",
            "description": "Get current stock price",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    }
                },
                "required": ["ticker"]
            }
        }
        
        Output:
        - yahoo_stock_price(ticker: string): Get current stock price
          Parameters:
            * ticker (required): Stock ticker symbol
    """
    tool_name = tool.get("name", "unknown")
    description = tool.get("description", "No description available")
    parameters = tool.get("parameters", {})
    
    # Extract parameter information
    properties = parameters.get("properties", {})
    required_params = parameters.get("required", [])
    
    # Build parameter signature for function header
    param_signature_parts = []
    for param_name, param_info in properties.items():
        param_type = param_info.get("type", "any")
        param_signature_parts.append(f"{param_name}: {param_type}")
    
    param_signature = ", ".join(param_signature_parts) if param_signature_parts else ""
    
    # Build the tool header line
    tool_header = f"- {tool_name}({param_signature}): {description}"
    
    # Build parameter details section
    if properties:
        param_details = ["  Parameters:"]
        for param_name, param_info in properties.items():
            param_desc = param_info.get("description", "No description")
            is_required = param_name in required_params
            requirement_label = "required" if is_required else "optional"
            param_details.append(f"    * {param_name} ({requirement_label}): {param_desc}")
        
        return tool_header + "\n" + "\n".join(param_details)
    else:
        # No parameters
        return tool_header + "\n  Parameters: None"


def format_all_tools(tools: List[Dict[str, Any]]) -> str:
    """
    Format multiple tool schemas into a single tool instruction block.
    
    Accepts a list of LangChain tool definitions, formats each using format_tool_schema,
    and combines them into a single structured block that can be included in the prompt.
    
    Args:
        tools: List of LangChain tool definition dictionaries
    
    Returns:
        Combined tool instruction string with all tool schemas formatted consistently
    
    Example:
        Input:
        [
            {
                "name": "yahoo_stock_price",
                "description": "Get current stock price",
                "parameters": {...}
            },
            {
                "name": "yahoo_news",
                "description": "Get latest news",
                "parameters": {...}
            }
        ]
        
        Output:
        - yahoo_stock_price(ticker: string): Get current stock price
          Parameters:
            * ticker (required): Stock ticker symbol
        
        - yahoo_news(ticker: string, num_articles: number): Get latest news
          Parameters:
            * ticker (required): Stock ticker symbol
            * num_articles (optional): Number of articles to retrieve
    """
    if not tools:
        return ""
    
    # Format each tool schema
    formatted_schemas = []
    for tool in tools:
        formatted_schema = format_tool_schema(tool)
        formatted_schemas.append(formatted_schema)
    
    # Combine with blank lines between tools for readability
    return "\n\n".join(formatted_schemas)


def build_tool_instruction(tools: List[Dict[str, Any]]) -> str:
    """
    Build complete tool instruction block using the TOOL_INSTRUCTION_TEMPLATE.
    
    This function combines the tool schema formatting with the instruction template
    to create a complete prompt section that instructs the model on how to use tools.
    
    Args:
        tools: List of LangChain tool definition dictionaries
    
    Returns:
        Complete tool instruction string ready to be injected into the system prompt.
        Returns empty string if no tools are provided.
    
    Example:
        Input:
        [
            {
                "name": "yahoo_stock_price",
                "description": "Get current stock price",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Stock ticker symbol"}
                    },
                    "required": ["ticker"]
                }
            }
        ]
        
        Output:
        You have access to the following tools:

        - yahoo_stock_price(ticker: string): Get current stock price
          Parameters:
            * ticker (required): Stock ticker symbol

        To use a tool, respond with a JSON object in this EXACT format:
        {"tool": "tool_name", "args": {"param1": "value1", "param2": "value2"}}

        Only use this format when you need to call a tool. Otherwise, respond normally.
        After receiving tool results, provide a comprehensive answer based on the data.
    """
    if not tools:
        return ""
    
    # Format all tool schemas
    tool_schemas = format_all_tools(tools)
    
    # Inject into template
    return TOOL_INSTRUCTION_TEMPLATE.format(tool_schemas=tool_schemas)
