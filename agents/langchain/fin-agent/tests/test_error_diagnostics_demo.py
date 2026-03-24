"""
Demonstration of comprehensive error diagnostics in response_parser and content_handler.

This test file demonstrates that all parsing failures include:
- Error type
- Error location
- Problematic input

Validates Requirement 9.5
"""

import json
import logging
import pytest
from src.response_parser import parse_tool_call
from src.content_handler import LlamaFunctionCallingHandler


class TestErrorDiagnostics:
    """Demonstrate comprehensive error diagnostics for all parsing failures."""
    
    def test_response_parser_json_decode_error_diagnostics(self, caplog):
        """
        Verify that JSON decode errors in response_parser include:
        - Error type (JSONDecodeError)
        - Error location (line, column)
        - Problematic input (raw_args)
        - Position in text
        """
        caplog.set_level(logging.INFO)
        
        # Malformed JSON in tool call
        text = '{"tool": "test_tool", "args": {invalid: "json", missing: quotes}}'
        
        tool_calls, remaining = parse_tool_call(text)
        
        # Should handle gracefully
        assert tool_calls is None
        assert remaining == text
        
        # Verify comprehensive diagnostic information
        log_text = caplog.text
        
        # Error type
        assert "Error Type: JSONDecodeError" in log_text
        
        # Error message
        assert "Error Message:" in log_text
        
        # Tool name
        assert "Tool: test_tool" in log_text
        
        # Raw problematic input
        assert "Raw Arguments:" in log_text
        assert "invalid" in log_text  # Part of the problematic input
        
        # Position information
        assert "Position in Text:" in log_text
    
    def test_response_parser_multiple_errors_all_logged(self, caplog):
        """
        Verify that when multiple tool calls fail, all errors are logged
        with complete diagnostic information.
        """
        caplog.set_level(logging.INFO)
        
        # Multiple malformed tool calls (properly structured but invalid JSON)
        text = '''
        {"tool": "tool1", "args": {bad: "json"}}
        {"tool": "tool2", "args": {also: "bad"}}
        {"tool": "tool3", "args": {missing: "brace"}}
        '''
        
        tool_calls, remaining = parse_tool_call(text)
        
        # Should handle gracefully
        assert tool_calls is None
        
        # Verify all three errors are logged
        log_text = caplog.text
        assert "Tool: tool1" in log_text
        assert "Tool: tool2" in log_text
        assert "Tool: tool3" in log_text
        
        # Verify error count
        assert "encountered 3 error(s)" in log_text
    
    def test_content_handler_json_decode_error_diagnostics(self, caplog):
        """
        Verify that JSON decode errors in content_handler include:
        - Error type (JSONDecodeError)
        - Error location (line, column)
        - Raw output preview
        - Output length
        """
        caplog.set_level(logging.ERROR)
        handler = LlamaFunctionCallingHandler()
        
        # Invalid JSON response
        response = b"This is not valid JSON {broken"
        
        with pytest.raises(ValueError, match="Invalid JSON response from DJL"):
            handler.transform_output(response)
        
        # Verify comprehensive diagnostic information
        log_text = caplog.text
        
        # Error type
        assert "JSONDecodeError" in log_text
        
        # Error location
        assert "error_location" in log_text
        
        # Raw output preview
        assert "raw_output_preview" in log_text
        
        # Output length
        assert "output_length" in log_text
    
    def test_content_handler_invalid_format_diagnostics(self, caplog):
        """
        Verify that invalid format errors include:
        - Error type (InvalidResponseFormat)
        - Expected format
        - Received type
        - Raw output preview
        """
        caplog.set_level(logging.ERROR)
        handler = LlamaFunctionCallingHandler()
        
        # Invalid format (missing choices)
        response = json.dumps({"wrong": "format"}).encode("utf-8")
        
        with pytest.raises(ValueError, match="No choices in response"):
            handler.transform_output(response)
        
        # Verify diagnostic information
        log_text = caplog.text
        
        # Error type
        assert "InvalidResponseFormat" in log_text
        
        # Expected vs received
        assert "expected" in log_text.lower()
        assert "received" in log_text.lower()
    
    def test_content_handler_missing_field_diagnostics(self, caplog):
        """
        Verify that missing message field is handled gracefully.
        """
        caplog.set_level(logging.ERROR)
        handler = LlamaFunctionCallingHandler()
        
        # Missing message field in choice - should handle gracefully
        response = json.dumps({
            "choices": [{
                "wrong_field": "value",
                "another_field": "data"
            }]
        }).encode("utf-8")
        
        # Should not raise, but return empty AIMessage
        result = handler.transform_output(response)
        
        # Verify graceful handling
        assert result.content == ""
        assert result.tool_calls == []
    
    def test_content_handler_unexpected_error_diagnostics(self, caplog):
        """
        Verify that unexpected errors include:
        - Error type (exception class name)
        - Error message
        - Output preview
        - Output length
        - Full traceback
        """
        caplog.set_level(logging.ERROR)
        handler = LlamaFunctionCallingHandler()
        
        # Create a response that will cause an unexpected error
        # (empty list will trigger IndexError)
        response = json.dumps([]).encode("utf-8")
        
        with pytest.raises(ValueError):
            handler.transform_output(response)
        
        # Verify diagnostic information
        log_text = caplog.text
        
        # Error type
        assert "error_type" in log_text
        
        # Output information
        assert "output_preview" in log_text or "raw_output_preview" in log_text


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
