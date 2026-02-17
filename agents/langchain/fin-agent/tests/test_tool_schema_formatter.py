"""
Unit tests for tool schema formatter.
"""

import pytest
from src.tool_schema_formatter import format_tool_schema


def test_format_tool_schema_basic():
    """Test basic tool schema formatting with required parameters."""
    tool = {
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
    
    result = format_tool_schema(tool)
    
    # Check that all key components are present
    assert "yahoo_stock_price" in result
    assert "ticker: string" in result
    assert "Get current stock price" in result
    assert "ticker (required)" in result
    assert "Stock ticker symbol" in result


def test_format_tool_schema_multiple_params():
    """Test tool schema with multiple parameters, some optional."""
    tool = {
        "name": "search_news",
        "description": "Search for news articles",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results"
                },
                "date_from": {
                    "type": "string",
                    "description": "Start date for search"
                }
            },
            "required": ["query"]
        }
    }
    
    result = format_tool_schema(tool)
    
    # Check required parameter
    assert "query (required)" in result
    assert "Search query" in result
    
    # Check optional parameters
    assert "limit (optional)" in result
    assert "date_from (optional)" in result
    assert "Maximum number of results" in result


def test_format_tool_schema_no_params():
    """Test tool schema with no parameters."""
    tool = {
        "name": "get_current_time",
        "description": "Get the current time",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
    
    result = format_tool_schema(tool)
    
    assert "get_current_time()" in result
    assert "Get the current time" in result
    assert "Parameters: None" in result


def test_format_tool_schema_missing_description():
    """Test tool schema with missing parameter descriptions."""
    tool = {
        "name": "test_tool",
        "description": "A test tool",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string"
                    # No description
                }
            },
            "required": ["param1"]
        }
    }
    
    result = format_tool_schema(tool)
    
    assert "test_tool" in result
    assert "param1 (required)" in result
    assert "No description" in result


def test_format_tool_schema_all_required():
    """Test tool schema where all parameters are required."""
    tool = {
        "name": "calculate",
        "description": "Perform calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "First number"},
                "y": {"type": "number", "description": "Second number"},
                "operation": {"type": "string", "description": "Operation to perform"}
            },
            "required": ["x", "y", "operation"]
        }
    }
    
    result = format_tool_schema(tool)
    
    # All should be marked as required
    assert "x (required)" in result
    assert "y (required)" in result
    assert "operation (required)" in result
    
    # No optional parameters
    assert "(optional)" not in result


def test_format_tool_schema_structure():
    """Test that the formatted schema has the expected structure."""
    tool = {
        "name": "example_tool",
        "description": "An example tool",
        "parameters": {
            "type": "object",
            "properties": {
                "arg1": {"type": "string", "description": "First argument"}
            },
            "required": ["arg1"]
        }
    }
    
    result = format_tool_schema(tool)
    lines = result.split("\n")
    
    # First line should be the tool header with signature
    assert lines[0].startswith("- example_tool(")
    assert "arg1: string" in lines[0]
    assert "An example tool" in lines[0]
    
    # Second line should be "  Parameters:"
    assert lines[1].strip() == "Parameters:"
    
    # Third line should be the parameter detail
    assert "* arg1 (required)" in lines[2]



def test_format_all_tools_single_tool():
    """Test format_all_tools with a single tool."""
    from src.tool_schema_formatter import format_all_tools
    
    tools = [
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
    ]
    
    result = format_all_tools(tools)
    
    # Should contain the tool schema
    assert "yahoo_stock_price" in result
    assert "ticker: string" in result
    assert "Get current stock price" in result


def test_format_all_tools_multiple_tools():
    """Test format_all_tools with multiple tools."""
    from src.tool_schema_formatter import format_all_tools
    
    tools = [
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
        },
        {
            "name": "yahoo_news",
            "description": "Get latest news",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "num_articles": {"type": "integer", "description": "Number of articles"}
                },
                "required": ["ticker"]
            }
        }
    ]
    
    result = format_all_tools(tools)
    
    # Should contain both tool schemas
    assert "yahoo_stock_price" in result
    assert "yahoo_news" in result
    assert "Get current stock price" in result
    assert "Get latest news" in result
    
    # Should have proper separation between tools (blank line)
    assert "\n\n" in result


def test_format_all_tools_empty_list():
    """Test format_all_tools with empty list."""
    from src.tool_schema_formatter import format_all_tools
    
    result = format_all_tools([])
    
    # Should return empty string
    assert result == ""


def test_format_all_tools_preserves_order():
    """Test that format_all_tools preserves the order of tools."""
    from src.tool_schema_formatter import format_all_tools
    
    tools = [
        {
            "name": "tool_a",
            "description": "First tool",
            "parameters": {"type": "object", "properties": {}}
        },
        {
            "name": "tool_b",
            "description": "Second tool",
            "parameters": {"type": "object", "properties": {}}
        },
        {
            "name": "tool_c",
            "description": "Third tool",
            "parameters": {"type": "object", "properties": {}}
        }
    ]
    
    result = format_all_tools(tools)
    
    # Find positions of each tool name
    pos_a = result.find("tool_a")
    pos_b = result.find("tool_b")
    pos_c = result.find("tool_c")
    
    # Verify order is preserved
    assert pos_a < pos_b < pos_c


def test_format_all_tools_consistent_structure():
    """Test that all tools in format_all_tools have consistent structure."""
    from src.tool_schema_formatter import format_all_tools
    
    tools = [
        {
            "name": "tool1",
            "description": "Tool one",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "Parameter 1"}
                },
                "required": ["param1"]
            }
        },
        {
            "name": "tool2",
            "description": "Tool two",
            "parameters": {
                "type": "object",
                "properties": {
                    "param2": {"type": "number", "description": "Parameter 2"}
                },
                "required": []
            }
        }
    ]
    
    result = format_all_tools(tools)
    
    # Split by double newline to get individual tool schemas
    tool_schemas = result.split("\n\n")
    
    assert len(tool_schemas) == 2
    
    # Each schema should start with "- tool_name"
    assert tool_schemas[0].startswith("- tool1")
    assert tool_schemas[1].startswith("- tool2")
    
    # Each schema should contain "Parameters:"
    assert "Parameters:" in tool_schemas[0]
    assert "Parameters:" in tool_schemas[1]


def test_tool_instruction_template_exists():
    """Test that TOOL_INSTRUCTION_TEMPLATE constant is defined."""
    from src.tool_schema_formatter import TOOL_INSTRUCTION_TEMPLATE
    
    # Should be a non-empty string
    assert isinstance(TOOL_INSTRUCTION_TEMPLATE, str)
    assert len(TOOL_INSTRUCTION_TEMPLATE) > 0


def test_tool_instruction_template_has_placeholder():
    """Test that TOOL_INSTRUCTION_TEMPLATE has the tool_schemas placeholder."""
    from src.tool_schema_formatter import TOOL_INSTRUCTION_TEMPLATE
    
    # Should contain the {tool_schemas} placeholder
    assert "{tool_schemas}" in TOOL_INSTRUCTION_TEMPLATE


def test_tool_instruction_template_has_json_format():
    """Test that TOOL_INSTRUCTION_TEMPLATE includes JSON format specification."""
    from src.tool_schema_formatter import TOOL_INSTRUCTION_TEMPLATE
    
    # Should specify the JSON format for tool calls
    assert '"tool"' in TOOL_INSTRUCTION_TEMPLATE
    assert '"args"' in TOOL_INSTRUCTION_TEMPLATE
    assert '{"tool":' in TOOL_INSTRUCTION_TEMPLATE or '{"tool" :' in TOOL_INSTRUCTION_TEMPLATE


def test_tool_instruction_template_has_usage_guidelines():
    """Test that TOOL_INSTRUCTION_TEMPLATE includes usage guidelines."""
    from src.tool_schema_formatter import TOOL_INSTRUCTION_TEMPLATE
    
    # Should include guidance on when to use tools
    template_lower = TOOL_INSTRUCTION_TEMPLATE.lower()
    assert "tool" in template_lower
    assert any(word in template_lower for word in ["use", "call", "respond"])


def test_build_tool_instruction_single_tool():
    """Test build_tool_instruction with a single tool."""
    from src.tool_schema_formatter import build_tool_instruction
    
    tools = [
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
    
    result = build_tool_instruction(tools)
    
    # Should contain the tool schema
    assert "yahoo_stock_price" in result
    assert "ticker: string" in result
    
    # Should contain the JSON format specification
    assert '"tool"' in result
    assert '"args"' in result
    
    # Should contain usage guidelines
    assert "tool" in result.lower()


def test_build_tool_instruction_multiple_tools():
    """Test build_tool_instruction with multiple tools."""
    from src.tool_schema_formatter import build_tool_instruction
    
    tools = [
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
        },
        {
            "name": "yahoo_news",
            "description": "Get latest news",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["ticker"]
            }
        }
    ]
    
    result = build_tool_instruction(tools)
    
    # Should contain both tool schemas
    assert "yahoo_stock_price" in result
    assert "yahoo_news" in result
    
    # Should contain the template instructions
    assert '"tool"' in result
    assert '"args"' in result


def test_build_tool_instruction_empty_list():
    """Test build_tool_instruction with empty list."""
    from src.tool_schema_formatter import build_tool_instruction
    
    result = build_tool_instruction([])
    
    # Should return empty string
    assert result == ""


def test_build_tool_instruction_complete_structure():
    """Test that build_tool_instruction produces a complete instruction block."""
    from src.tool_schema_formatter import build_tool_instruction
    
    tools = [
        {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "Test parameter"}
                },
                "required": ["param1"]
            }
        }
    ]
    
    result = build_tool_instruction(tools)
    
    # Should have the tool schema section
    assert "test_tool" in result
    assert "param1: string" in result
    
    # Should have the JSON format example
    assert '{"tool":' in result or '{"tool" :' in result
    
    # Should have clear instructions
    result_lower = result.lower()
    assert "format" in result_lower or "use" in result_lower


def test_build_tool_instruction_json_format_example():
    """Test that build_tool_instruction includes a valid JSON format example."""
    from src.tool_schema_formatter import build_tool_instruction
    
    tools = [
        {
            "name": "example_tool",
            "description": "Example",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg": {"type": "string", "description": "Argument"}
                },
                "required": ["arg"]
            }
        }
    ]
    
    result = build_tool_instruction(tools)
    
    # Should show the structure with both "tool" and "args" keys
    assert '"tool"' in result
    assert '"args"' in result
    
    # Should show example parameter structure
    assert "param" in result.lower() or "value" in result.lower()
