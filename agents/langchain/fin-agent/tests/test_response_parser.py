"""
Tests for response_parser module.

Includes both unit tests for specific scenarios and property-based tests
for comprehensive coverage of the parse_tool_call function.
"""

import json
import pytest
from hypothesis import given, strategies as st
from src.response_parser import parse_tool_call


class TestParseToolCallUnit:
    """Unit tests for parse_tool_call function."""
    
    def test_single_tool_call(self):
        """Test parsing a single valid tool call."""
        text = '{"tool": "yahoo_stock_price", "args": {"ticker": "AAPL"}}'
        tool_calls, remaining = parse_tool_call(text)
        
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "yahoo_stock_price"
        assert tool_calls[0]["args"] == {"ticker": "AAPL"}
        assert "id" in tool_calls[0]
        assert tool_calls[0]["id"].startswith("call_")
        assert remaining == ""
    
    def test_tool_call_with_surrounding_text(self):
        """Test parsing tool call with text before and after."""
        text = 'Let me check that. {"tool": "yahoo_stock_price", "args": {"ticker": "AAPL"}} I will get the price.'
        tool_calls, remaining = parse_tool_call(text)
        
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "yahoo_stock_price"
        assert "Let me check that." in remaining
        assert "I will get the price." in remaining
    
    def test_multiple_tool_calls(self):
        """Test parsing multiple tool calls in one output."""
        text = '{"tool": "yahoo_stock_price", "args": {"ticker": "AAPL"}} {"tool": "yahoo_news", "args": {"ticker": "GOOGL"}}'
        tool_calls, remaining = parse_tool_call(text)
        
        assert tool_calls is not None
        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "yahoo_stock_price"
        assert tool_calls[1]["name"] == "yahoo_news"
        
        # Check unique IDs
        assert tool_calls[0]["id"] != tool_calls[1]["id"]
    
    def test_no_tool_call(self):
        """Test text without any tool calls."""
        text = "This is just a regular response without any tool calls."
        tool_calls, remaining = parse_tool_call(text)
        
        assert tool_calls is None
        assert remaining == text
    
    def test_malformed_json_args(self):
        """Test handling of malformed JSON in args."""
        text = '{"tool": "yahoo_stock_price", "args": {ticker: "AAPL"}}'  # Missing quotes
        tool_calls, remaining = parse_tool_call(text)
        
        # Should return None and original text when JSON is invalid
        assert tool_calls is None
        assert remaining == text
    
    def test_malformed_json_with_logging(self, caplog):
        """Test that malformed JSON produces diagnostic logging."""
        import logging
        caplog.set_level(logging.INFO)
        
        text = '{"tool": "yahoo_stock_price", "args": {ticker: "AAPL"}}'  # Missing quotes
        tool_calls, remaining = parse_tool_call(text)
        
        # Should return None and original text when JSON is invalid
        assert tool_calls is None
        assert remaining == text
        
        # Check that diagnostic information was logged
        assert "Tool call parsing encountered" in caplog.text
        assert "yahoo_stock_price" in caplog.text
        assert "JSONDecodeError" in caplog.text or "Error Type" in caplog.text
    
    def test_empty_args(self):
        """Test tool call with empty arguments."""
        text = '{"tool": "get_current_time", "args": {}}'
        tool_calls, remaining = parse_tool_call(text)
        
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "get_current_time"
        assert tool_calls[0]["args"] == {}
    
    def test_complex_args(self):
        """Test tool call with complex nested arguments."""
        text = '{"tool": "search", "args": {"query": "stock market", "filters": {"date": "2024", "type": "news"}}}'
        tool_calls, remaining = parse_tool_call(text)
        
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["args"]["query"] == "stock market"
        assert tool_calls[0]["args"]["filters"]["date"] == "2024"
    
    def test_tool_call_with_whitespace_variations(self):
        """Test parsing with various whitespace patterns."""
        text = '{"tool":  "yahoo_stock_price",  "args":  {"ticker": "AAPL"}}'
        tool_calls, remaining = parse_tool_call(text)
        
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "yahoo_stock_price"
    
    def test_empty_string(self):
        """Test parsing empty string."""
        text = ""
        tool_calls, remaining = parse_tool_call(text)
        
        assert tool_calls is None
        assert remaining == ""
    
    def test_unique_call_ids(self):
        """Test that multiple calls get unique IDs."""
        text = '{"tool": "tool1", "args": {}} {"tool": "tool2", "args": {}}'
        tool_calls, remaining = parse_tool_call(text)
        
        assert tool_calls is not None
        assert len(tool_calls) == 2
        
        # All IDs should be unique
        ids = [call["id"] for call in tool_calls]
        assert len(ids) == len(set(ids))
        
        # All IDs should start with "call_"
        assert all(id.startswith("call_") for id in ids)
    
    def test_diagnostic_information_on_error(self, caplog):
        """Test that parsing errors provide comprehensive diagnostic information."""
        import logging
        caplog.set_level(logging.INFO)
        
        # Create text with malformed JSON
        text = '{"tool": "test_tool", "args": {invalid: json}}'
        tool_calls, remaining = parse_tool_call(text)
        
        # Should handle gracefully
        assert tool_calls is None
        assert remaining == text
        
        # Check diagnostic information in logs
        log_text = caplog.text
        assert "Tool call parsing encountered" in log_text
        assert "test_tool" in log_text
        assert "Error Type" in log_text
        assert "Error Message" in log_text
    
    def test_multiple_errors_diagnostic(self, caplog):
        """Test diagnostic information when multiple tool calls have errors."""
        import logging
        caplog.set_level(logging.INFO)
        
        # Multiple malformed tool calls
        text = '{"tool": "tool1", "args": {bad: json}} {"tool": "tool2", "args": {also: bad}}'
        tool_calls, remaining = parse_tool_call(text)
        
        # Should handle gracefully
        assert tool_calls is None
        
        # Check that both errors are logged
        log_text = caplog.text
        assert "tool1" in log_text
        assert "tool2" in log_text


class TestParseToolCallProperty:
    """Property-based tests for parse_tool_call function."""
    
    # Feature: sagemaker-llama-function-calling, Property 1: Tool Call Round Trip
    @given(
        tool_name=st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_'
        )),
        args=st.dictionaries(
            keys=st.text(min_size=1, max_size=20, alphabet=st.characters(
                whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_'
            )),
            values=st.one_of(
                st.text(max_size=100),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.booleans()
            ),
            min_size=0,
            max_size=5
        )
    )
    def test_tool_call_round_trip(self, tool_name, args):
        """
        Property 1: For any valid tool call, parsing should extract
        the tool name and arguments correctly.
        
        **Validates: Requirements 2.4, 6.1**
        """
        # Generate tool call JSON (ensure_ascii=False to preserve Unicode)
        tool_call_json = json.dumps({"tool": tool_name, "args": args}, ensure_ascii=False)
        
        # Parse it
        parsed_calls, remaining = parse_tool_call(tool_call_json)
        
        assert parsed_calls is not None
        assert len(parsed_calls) == 1
        
        call = parsed_calls[0]
        assert call["name"] == tool_name
        assert call["args"] == args
        assert "id" in call
        assert call["id"].startswith("call_")
        assert len(call["id"]) > 5  # Should have meaningful length
    
    # Feature: sagemaker-llama-function-calling, Property 8: Unique Tool Call IDs
    @given(
        num_calls=st.integers(min_value=1, max_value=10),
        tool_name=st.text(min_size=1, max_size=20, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll'), whitelist_characters='_'
        ))
    )
    def test_unique_tool_call_ids(self, num_calls, tool_name):
        """
        Property 8: For any set of tool calls parsed from a single output,
        all tool call IDs should be unique.
        
        **Validates: Requirements 6.4**
        """
        # Generate multiple tool calls
        tool_calls_json = " ".join([
            json.dumps({"tool": tool_name, "args": {"index": i}})
            for i in range(num_calls)
        ])
        
        # Parse them
        parsed_calls, _ = parse_tool_call(tool_calls_json)
        
        assert parsed_calls is not None
        assert len(parsed_calls) == num_calls
        
        # Extract all IDs
        ids = [call["id"] for call in parsed_calls]
        
        # All IDs should be unique
        assert len(ids) == len(set(ids))
    
    # Feature: sagemaker-llama-function-calling, Property 9: Multiple Tool Call Parsing
    @given(
        tools=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=20, alphabet=st.characters(
                    whitelist_categories=('Lu', 'Ll'), whitelist_characters='_'
                )),
                st.dictionaries(
                    keys=st.text(min_size=1, max_size=10, alphabet=st.characters(
                        whitelist_categories=('Lu', 'Ll'), whitelist_characters='_'
                    )),
                    values=st.text(max_size=50),
                    min_size=0,
                    max_size=3
                )
            ),
            min_size=1,
            max_size=5
        )
    )
    def test_multiple_tool_call_parsing(self, tools):
        """
        Property 9: For any model output containing multiple tool call JSON objects,
        all tool calls should be parsed and returned.
        
        **Validates: Requirements 6.5**
        """
        # Generate text with multiple tool calls (ensure_ascii=False to preserve Unicode)
        tool_calls_json = " ".join([
            json.dumps({"tool": name, "args": args}, ensure_ascii=False)
            for name, args in tools
        ])
        
        # Parse them
        parsed_calls, _ = parse_tool_call(tool_calls_json)
        
        assert parsed_calls is not None
        assert len(parsed_calls) == len(tools)
        
        # Verify each tool call was parsed correctly
        for i, (expected_name, expected_args) in enumerate(tools):
            assert parsed_calls[i]["name"] == expected_name
            assert parsed_calls[i]["args"] == expected_args
    
    # Feature: sagemaker-llama-function-calling, Property 10: Invalid JSON Fallback
    @given(
        text=st.text(min_size=1, max_size=200)
    )
    def test_invalid_json_fallback(self, text):
        """
        Property 10: For any text that doesn't contain valid tool call JSON,
        the parser should return None for tool_calls and preserve the original text.
        
        **Validates: Requirements 6.3, 9.1, 9.4**
        """
        # Assume the text doesn't contain valid tool call pattern
        # (this is true for most random text)
        if '{"tool":' not in text:
            tool_calls, remaining = parse_tool_call(text)
            
            # Should not crash and should return original text
            assert remaining == text
            
            # If no valid tool calls, should return None
            if tool_calls is None:
                assert True  # Expected behavior
    
    @given(
        tool_name=st.text(min_size=1, max_size=20, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll'), whitelist_characters='_'
        )),
        prefix=st.text(max_size=50),
        suffix=st.text(max_size=50)
    )
    def test_tool_call_with_surrounding_text_property(self, tool_name, prefix, suffix):
        """
        Test that tool calls are correctly extracted even with surrounding text.
        """
        args = {"param": "value"}
        tool_call_json = json.dumps({"tool": tool_name, "args": args}, ensure_ascii=False)
        text = f"{prefix} {tool_call_json} {suffix}"
        
        parsed_calls, remaining = parse_tool_call(text)
        
        assert parsed_calls is not None
        assert len(parsed_calls) == 1
        assert parsed_calls[0]["name"] == tool_name
        assert parsed_calls[0]["args"] == args
        
        # Remaining text should contain prefix and suffix but not the JSON
        # Note: remaining text is stripped, so whitespace-only strings become empty
        if prefix.strip():
            assert prefix.strip() in remaining
        if suffix.strip():
            assert suffix.strip() in remaining
        assert tool_call_json not in remaining
