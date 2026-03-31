"""
Unit tests for prompt_formatter module.

Tests the build_llama3_prompt function to ensure proper Llama 3 chat template formatting.
"""

import pytest
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from src.prompt_formatter import build_llama3_prompt


def test_build_llama3_prompt_starts_with_begin_of_text():
    """Test that prompt starts with <|begin_of_text|> token."""
    messages = [HumanMessage(content="Hello")]
    prompt = build_llama3_prompt(messages)
    assert prompt.startswith("<|begin_of_text|>")


def test_build_llama3_prompt_ends_with_assistant_header():
    """Test that prompt ends with assistant header to prompt response."""
    messages = [HumanMessage(content="Hello")]
    prompt = build_llama3_prompt(messages)
    assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


def test_build_llama3_prompt_formats_human_message():
    """Test that HumanMessage is formatted with user role."""
    messages = [HumanMessage(content="What is the weather?")]
    prompt = build_llama3_prompt(messages)
    
    assert "<|start_header_id|>user<|end_header_id|>" in prompt
    assert "What is the weather?" in prompt
    assert "<|eot_id|>" in prompt


def test_build_llama3_prompt_formats_ai_message():
    """Test that AIMessage is formatted with assistant role."""
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
    ]
    prompt = build_llama3_prompt(messages)
    
    assert "<|start_header_id|>assistant<|end_header_id|>" in prompt
    assert "Hi there!" in prompt
    # Should have 2 eot_id tokens (one for each message)
    assert prompt.count("<|eot_id|>") == 2


def test_build_llama3_prompt_formats_system_message():
    """Test that SystemMessage is formatted with system role."""
    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content="Hello"),
    ]
    prompt = build_llama3_prompt(messages)
    
    assert "<|start_header_id|>system<|end_header_id|>" in prompt
    assert "You are a helpful assistant" in prompt


def test_build_llama3_prompt_formats_tool_message():
    """Test that ToolMessage is formatted as user message with tool result."""
    messages = [
        HumanMessage(content="What is AAPL price?"),
        ToolMessage(content="$150.25", tool_call_id="call_123", name="yahoo_stock_price"),
    ]
    prompt = build_llama3_prompt(messages)
    
    # Tool message should be formatted as user message
    assert prompt.count("<|start_header_id|>user<|end_header_id|>") == 2
    assert "Tool result from yahoo_stock_price:" in prompt
    assert "$150.25" in prompt


def test_build_llama3_prompt_injects_tool_instruction_into_system_message():
    """Test that tool instruction is injected into existing system message."""
    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content="Hello"),
    ]
    tool_instruction = "You have access to tools: tool1, tool2"
    prompt = build_llama3_prompt(messages, tool_instruction)
    
    assert "You are a helpful assistant" in prompt
    assert "You have access to tools: tool1, tool2" in prompt
    # Should appear in the same system message block
    system_start = prompt.find("<|start_header_id|>system<|end_header_id|>")
    system_end = prompt.find("<|eot_id|>", system_start)
    system_block = prompt[system_start:system_end]
    assert "You are a helpful assistant" in system_block
    assert "You have access to tools: tool1, tool2" in system_block


def test_build_llama3_prompt_creates_system_message_for_tool_instruction():
    """Test that system message is created if only tool instruction provided."""
    messages = [HumanMessage(content="Hello")]
    tool_instruction = "You have access to tools: tool1, tool2"
    prompt = build_llama3_prompt(messages, tool_instruction)
    
    assert "<|start_header_id|>system<|end_header_id|>" in prompt
    assert "You have access to tools: tool1, tool2" in prompt
    # System message should come before user message
    system_pos = prompt.find("<|start_header_id|>system<|end_header_id|>")
    user_pos = prompt.find("<|start_header_id|>user<|end_header_id|>")
    assert system_pos < user_pos


def test_build_llama3_prompt_handles_multiple_messages():
    """Test that multiple messages are formatted correctly in sequence."""
    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content="What is 2+2?"),
        AIMessage(content="4"),
        HumanMessage(content="What is 3+3?"),
    ]
    prompt = build_llama3_prompt(messages)
    
    # Check all messages are present
    assert "You are a helpful assistant" in prompt
    assert "What is 2+2?" in prompt
    assert "4" in prompt
    assert "What is 3+3?" in prompt
    
    # Check proper number of eot_id tokens (one per message)
    assert prompt.count("<|eot_id|>") == 4
    
    # Check message order
    system_pos = prompt.find("You are a helpful assistant")
    first_q_pos = prompt.find("What is 2+2?")
    first_a_pos = prompt.find("4")
    second_q_pos = prompt.find("What is 3+3?")
    
    assert system_pos < first_q_pos < first_a_pos < second_q_pos


def test_build_llama3_prompt_empty_messages():
    """Test that empty message list still produces valid prompt structure."""
    messages = []
    prompt = build_llama3_prompt(messages)
    
    assert prompt.startswith("<|begin_of_text|>")
    assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


def test_build_llama3_prompt_preserves_message_content():
    """Test that message content with special characters is preserved."""
    messages = [
        HumanMessage(content="Test with\nnewlines and\ttabs"),
        AIMessage(content='Test with "quotes" and {braces}'),
    ]
    prompt = build_llama3_prompt(messages)
    
    assert "Test with\nnewlines and\ttabs" in prompt
    assert 'Test with "quotes" and {braces}' in prompt


def test_build_llama3_prompt_tool_message_without_name():
    """Test that ToolMessage without name attribute still works."""
    messages = [
        HumanMessage(content="Query"),
        ToolMessage(content="Result", tool_call_id="call_123"),
    ]
    prompt = build_llama3_prompt(messages)
    
    # Should use default "tool" name
    assert "Tool result from tool:" in prompt
    assert "Result" in prompt


def test_build_llama3_prompt_with_tools_parameter():
    """Test that tools parameter formats and injects tool schemas."""
    messages = [HumanMessage(content="What is the weather?")]
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    ]
    prompt = build_llama3_prompt(messages, tools=tools)
    
    # Should contain tool instruction template
    assert "You have access to the following tools:" in prompt
    assert "get_weather" in prompt
    assert "Get weather information" in prompt
    assert "location" in prompt
    assert "City name" in prompt
    assert '{"tool": "tool_name", "args": {"param1": "value1"' in prompt


def test_build_llama3_prompt_with_tools_injects_into_system_message():
    """Test that tools are injected into existing system message."""
    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content="Hello"),
    ]
    tools = [
        {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {"type": "object", "properties": {}}
        }
    ]
    prompt = build_llama3_prompt(messages, tools=tools)
    
    # Should have system message with both original content and tool instruction
    assert "You are a helpful assistant" in prompt
    assert "You have access to the following tools:" in prompt
    assert "test_tool" in prompt
    
    # Should be in the same system message block
    system_start = prompt.find("<|start_header_id|>system<|end_header_id|>")
    system_end = prompt.find("<|eot_id|>", system_start)
    system_block = prompt[system_start:system_end]
    assert "You are a helpful assistant" in system_block
    assert "test_tool" in system_block


def test_build_llama3_prompt_with_tools_creates_system_message():
    """Test that system message is created when only tools provided."""
    messages = [HumanMessage(content="Hello")]
    tools = [
        {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {"type": "object", "properties": {}}
        }
    ]
    prompt = build_llama3_prompt(messages, tools=tools)
    
    # Should create a system message with tool instruction
    assert "<|start_header_id|>system<|end_header_id|>" in prompt
    assert "You have access to the following tools:" in prompt
    assert "test_tool" in prompt


def test_build_llama3_prompt_with_multiple_tools():
    """Test that multiple tools are all included in the prompt."""
    messages = [HumanMessage(content="Hello")]
    tools = [
        {
            "name": "tool1",
            "description": "First tool",
            "parameters": {"type": "object", "properties": {}}
        },
        {
            "name": "tool2",
            "description": "Second tool",
            "parameters": {"type": "object", "properties": {}}
        },
        {
            "name": "tool3",
            "description": "Third tool",
            "parameters": {"type": "object", "properties": {}}
        }
    ]
    prompt = build_llama3_prompt(messages, tools=tools)
    
    # All tools should be present
    assert "tool1" in prompt
    assert "First tool" in prompt
    assert "tool2" in prompt
    assert "Second tool" in prompt
    assert "tool3" in prompt
    assert "Third tool" in prompt


def test_build_llama3_prompt_tool_instruction_takes_precedence():
    """Test that tool_instruction parameter takes precedence over tools parameter."""
    messages = [HumanMessage(content="Hello")]
    tools = [
        {
            "name": "tool_from_list",
            "description": "This should not appear",
            "parameters": {"type": "object", "properties": {}}
        }
    ]
    tool_instruction = "Custom tool instruction"
    prompt = build_llama3_prompt(messages, tool_instruction=tool_instruction, tools=tools)
    
    # Should use tool_instruction, not format tools
    assert "Custom tool instruction" in prompt
    assert "tool_from_list" not in prompt
    assert "This should not appear" not in prompt


def test_build_llama3_prompt_with_empty_tools_list():
    """Test that empty tools list doesn't add tool instruction."""
    messages = [HumanMessage(content="Hello")]
    tools = []
    prompt = build_llama3_prompt(messages, tools=tools)
    
    # Should not contain tool instruction
    assert "You have access to the following tools:" not in prompt


def test_build_llama3_prompt_with_none_tools():
    """Test that None tools parameter works correctly."""
    messages = [HumanMessage(content="Hello")]
    prompt = build_llama3_prompt(messages, tools=None)
    
    # Should not contain tool instruction
    assert "You have access to the following tools:" not in prompt
    # Should still be valid prompt
    assert prompt.startswith("<|begin_of_text|>")
    assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")



def test_build_llama3_prompt_graceful_degradation_on_tool_formatting_error(caplog, monkeypatch):
    """Test that tool schema formatting errors are handled gracefully."""
    import logging
    from src import prompt_formatter
    
    caplog.set_level(logging.WARNING)
    
    # Mock build_tool_instruction to raise an exception
    def mock_build_tool_instruction(tools):
        raise ValueError("Simulated tool formatting error")
    
    # Patch it in the prompt_formatter module where it's imported
    monkeypatch.setattr(prompt_formatter, "build_tool_instruction", mock_build_tool_instruction)
    
    messages = [HumanMessage(content="What is the weather?")]
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {"type": "object", "properties": {}}
        }
    ]
    
    # Should not raise an exception
    prompt = build_llama3_prompt(messages, tools=tools)
    
    # Should still produce a valid prompt without tool instruction
    assert prompt.startswith("<|begin_of_text|>")
    assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    assert "What is the weather?" in prompt
    
    # Should NOT contain tool instruction
    assert "You have access to the following tools:" not in prompt
    assert "get_weather" not in prompt
    
    # Should log warning with diagnostic information
    assert "Tool schema formatting failed" in caplog.text
    assert "falling back to basic instruction following" in caplog.text
    assert "ValueError" in caplog.text
    assert "Simulated tool formatting error" in caplog.text
    assert "Tool count: 1" in caplog.text


def test_build_llama3_prompt_graceful_degradation_with_system_message(caplog, monkeypatch):
    """Test graceful degradation preserves system message when tool formatting fails."""
    import logging
    from src import prompt_formatter
    
    caplog.set_level(logging.WARNING)
    
    # Mock build_tool_instruction to raise an exception
    def mock_build_tool_instruction(tools):
        raise RuntimeError("Tool formatting failed")
    
    # Patch it in the prompt_formatter module where it's imported
    monkeypatch.setattr(prompt_formatter, "build_tool_instruction", mock_build_tool_instruction)
    
    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content="Hello"),
    ]
    tools = [{"name": "test_tool", "description": "Test", "parameters": {}}]
    
    # Should not raise an exception
    prompt = build_llama3_prompt(messages, tools=tools)
    
    # Should preserve system message
    assert "You are a helpful assistant" in prompt
    assert "<|start_header_id|>system<|end_header_id|>" in prompt
    
    # Should NOT contain tool instruction
    assert "You have access to the following tools:" not in prompt
    assert "test_tool" not in prompt
    
    # Should log warning
    assert "Tool schema formatting failed" in caplog.text
    assert "RuntimeError" in caplog.text


def test_build_llama3_prompt_graceful_degradation_diagnostic_info(caplog, monkeypatch):
    """Test that diagnostic information is logged on tool formatting failure."""
    import logging
    from src import prompt_formatter
    
    caplog.set_level(logging.WARNING)
    
    # Mock build_tool_instruction to raise a custom exception
    def mock_build_tool_instruction(tools):
        raise KeyError("missing_field")
    
    # Patch it in the prompt_formatter module where it's imported
    monkeypatch.setattr(prompt_formatter, "build_tool_instruction", mock_build_tool_instruction)
    
    messages = [HumanMessage(content="Test")]
    tools = [
        {"name": "tool1", "description": "Test 1", "parameters": {}},
        {"name": "tool2", "description": "Test 2", "parameters": {}},
    ]
    
    # Should not raise an exception
    prompt = build_llama3_prompt(messages, tools=tools)
    
    # Check diagnostic information in logs
    log_text = caplog.text
    assert "Tool schema formatting failed" in log_text
    assert "KeyError" in log_text
    assert "missing_field" in log_text
    assert "Tool count: 2" in log_text
    assert "error_type" in log_text
    assert "error_message" in log_text
    assert "tools_provided" in log_text


def test_build_llama3_prompt_no_degradation_when_tool_instruction_provided(caplog, monkeypatch):
    """Test that pre-formatted tool_instruction bypasses tool formatting and error handling."""
    import logging
    from src import prompt_formatter
    
    caplog.set_level(logging.WARNING)
    
    # Mock build_tool_instruction to raise an exception
    # This should NOT be called when tool_instruction is provided
    def mock_build_tool_instruction(tools):
        raise ValueError("This should not be called")
    
    # Patch it in the prompt_formatter module where it's imported
    monkeypatch.setattr(prompt_formatter, "build_tool_instruction", mock_build_tool_instruction)
    
    messages = [HumanMessage(content="Test")]
    tools = [{"name": "test_tool", "description": "Test", "parameters": {}}]
    tool_instruction = "Custom tool instruction"
    
    # Should not raise an exception
    prompt = build_llama3_prompt(messages, tool_instruction=tool_instruction, tools=tools)
    
    # Should use the provided tool_instruction
    assert "Custom tool instruction" in prompt
    
    # Should NOT log any warnings (build_tool_instruction was not called)
    assert "Tool schema formatting failed" not in caplog.text


def test_build_llama3_prompt_graceful_degradation_allows_agent_to_continue(monkeypatch):
    """Test that agent can still respond to queries when tool formatting fails."""
    from src import prompt_formatter
    
    # Mock build_tool_instruction to raise an exception
    def mock_build_tool_instruction(tools):
        raise Exception("Critical tool formatting error")
    
    # Patch it in the prompt_formatter module where it's imported
    monkeypatch.setattr(prompt_formatter, "build_tool_instruction", mock_build_tool_instruction)
    
    # Simulate a real agent query
    messages = [
        SystemMessage(content="You are a financial assistant"),
        HumanMessage(content="What is the current price of AAPL stock?"),
    ]
    tools = [
        {
            "name": "yahoo_stock_price",
            "description": "Get stock price",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["ticker"]
            }
        }
    ]
    
    # Should not crash
    prompt = build_llama3_prompt(messages, tools=tools)
    
    # Should produce a valid prompt that the model can respond to
    assert prompt.startswith("<|begin_of_text|>")
    assert "You are a financial assistant" in prompt
    assert "What is the current price of AAPL stock?" in prompt
    assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    
    # The agent can still respond, just without tool calling capability
    # This is better than crashing completely
    assert len(prompt) > 0
    assert "<|start_header_id|>user<|end_header_id|>" in prompt
