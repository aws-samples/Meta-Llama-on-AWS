"""
Unit tests for LlamaFunctionCallingHandler with DJL/vLLM OpenAI-compatible format.

Tests the transform_input and transform_output methods for DJL container with vLLM backend.
"""

import json
import pytest
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


class TestTransformInput:
    """Tests for the transform_input method with OpenAI-compatible format."""
    
    def test_transform_input_basic_message(self):
        """Test transform_input with a basic message without tools."""
        from src.content_handler import LlamaFunctionCallingHandler
        
        handler = LlamaFunctionCallingHandler()
        messages = [HumanMessage(content="What is the weather today?")]
        model_kwargs = {}
        
        result = handler.transform_input(messages, model_kwargs)
        
        # Verify result is bytes
        assert isinstance(result, bytes)
        
        # Parse the JSON payload
        payload = json.loads(result.decode("utf-8"))
        
        # Verify OpenAI-compatible structure
        assert "messages" in payload
        assert "max_tokens" in payload
        assert "temperature" in payload
        assert "top_p" in payload
        
        # Verify messages format
        messages_list = payload["messages"]
        assert len(messages_list) == 1
        assert messages_list[0]["role"] == "user"
        assert messages_list[0]["content"] == "What is the weather today?"
        
        # Verify parameters
        assert payload["max_tokens"] == 512
        assert payload["temperature"] == 0.1
        assert payload["top_p"] == 0.9
    
    def test_transform_input_with_system_message(self):
        """Test transform_input with system and user messages."""
        from src.content_handler import LlamaFunctionCallingHandler
        
        handler = LlamaFunctionCallingHandler()
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello!")
        ]
        model_kwargs = {}
        
        result = handler.transform_input(messages, model_kwargs)
        payload = json.loads(result.decode("utf-8"))
        messages_list = payload["messages"]
        
        # Verify both messages are present
        assert len(messages_list) == 2
        assert messages_list[0]["role"] == "system"
        assert messages_list[0]["content"] == "You are a helpful assistant."
        assert messages_list[1]["role"] == "user"
        assert messages_list[1]["content"] == "Hello!"
    
    def test_transform_input_with_tools(self):
        """Test transform_input with tools in model_kwargs."""
        from src.content_handler import LlamaFunctionCallingHandler
        
        handler = LlamaFunctionCallingHandler()
        messages = [HumanMessage(content="What is the price of AAPL?")]
        
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
        
        model_kwargs = {"tools": tools}
        
        result = handler.transform_input(messages, model_kwargs)
        payload = json.loads(result.decode("utf-8"))
        
        # Verify tools are in OpenAI format
        assert "tools" in payload
        assert len(payload["tools"]) == 1
        tool = payload["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "yahoo_stock_price"
        assert tool["function"]["description"] == "Get current stock price"
        assert "parameters" in tool["function"]
    
    def test_transform_input_custom_parameters(self):
        """Test transform_input with custom generation parameters."""
        from src.content_handler import LlamaFunctionCallingHandler
        
        handler = LlamaFunctionCallingHandler()
        messages = [HumanMessage(content="Test")]
        
        model_kwargs = {
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95
        }
        
        result = handler.transform_input(messages, model_kwargs)
        payload = json.loads(result.decode("utf-8"))
        
        # Verify custom parameters are used
        assert payload["max_tokens"] == 1024
        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.95
    
    def test_transform_input_multiple_messages(self):
        """Test transform_input with a conversation history."""
        from src.content_handler import LlamaFunctionCallingHandler
        
        handler = LlamaFunctionCallingHandler()
        messages = [
            SystemMessage(content="You are a financial assistant."),
            HumanMessage(content="What is AAPL?"),
            AIMessage(content="AAPL is Apple Inc."),
            HumanMessage(content="What's its price?")
        ]
        model_kwargs = {}
        
        result = handler.transform_input(messages, model_kwargs)
        payload = json.loads(result.decode("utf-8"))
        messages_list = payload["messages"]
        
        # Verify all messages are present in order
        assert len(messages_list) == 4
        assert messages_list[0]["role"] == "system"
        assert messages_list[1]["role"] == "user"
        assert messages_list[2]["role"] == "assistant"
        assert messages_list[3]["role"] == "user"


class TestTransformOutput:
    """Tests for the transform_output method with OpenAI-compatible format."""
    
    def test_transform_output_regular_response(self):
        """Test transform_output with a regular text response."""
        from src.content_handler import LlamaFunctionCallingHandler
        
        handler = LlamaFunctionCallingHandler()
        
        # Simulate DJL/vLLM OpenAI-compatible response format
        response = json.dumps({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The current price of Apple (AAPL) is $150.25"
                },
                "finish_reason": "stop"
            }]
        }).encode("utf-8")
        
        result = handler.transform_output(response)
        
        # Verify result is AIMessage
        assert isinstance(result, AIMessage)
        
        # Verify content
        assert result.content == "The current price of Apple (AAPL) is $150.25"
        
        # Verify no tool calls
        assert result.tool_calls == []
    
    def test_transform_output_tool_call(self):
        """Test transform_output with a tool call response."""
        from src.content_handler import LlamaFunctionCallingHandler
        
        handler = LlamaFunctionCallingHandler()
        
        # Simulate tool call response in OpenAI format
        response = json.dumps({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "yahoo_stock_price",
                            "arguments": '{"ticker": "AAPL"}'
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }).encode("utf-8")
        
        result = handler.transform_output(response)
        
        # Verify result is AIMessage
        assert isinstance(result, AIMessage)
        
        # Verify tool calls
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "yahoo_stock_price"
        assert result.tool_calls[0]["args"] == {"ticker": "AAPL"}
        assert result.tool_calls[0]["id"] == "call_abc123"
        
        # Content should be empty
        assert result.content == ""
    
    def test_transform_output_multiple_tool_calls(self):
        """Test transform_output with multiple tool calls."""
        from src.content_handler import LlamaFunctionCallingHandler
        
        handler = LlamaFunctionCallingHandler()
        
        # Simulate multiple tool calls
        response = json.dumps({
            "choices": [{
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
                                "name": "yahoo_news",
                                "arguments": '{"ticker": "AAPL"}'
                            }
                        }
                    ]
                }
            }]
        }).encode("utf-8")
        
        result = handler.transform_output(response)
        
        # Verify multiple tool calls
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "yahoo_stock_price"
        assert result.tool_calls[1]["name"] == "yahoo_news"
        
        # Verify unique IDs
        assert result.tool_calls[0]["id"] != result.tool_calls[1]["id"]
    
    def test_transform_output_tool_call_with_text(self):
        """Test transform_output with tool call and content."""
        from src.content_handler import LlamaFunctionCallingHandler
        
        handler = LlamaFunctionCallingHandler()
        
        # Simulate tool call with text content
        response = json.dumps({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Let me check that for you.",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "yahoo_stock_price",
                            "arguments": '{"ticker": "AAPL"}'
                        }
                    }]
                }
            }]
        }).encode("utf-8")
        
        result = handler.transform_output(response)
        
        # Verify tool call is extracted
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "yahoo_stock_price"
        
        # Verify content is preserved
        assert result.content == "Let me check that for you."
    
    def test_transform_output_invalid_json_response(self):
        """Test transform_output with invalid JSON response."""
        from src.content_handler import LlamaFunctionCallingHandler
        
        handler = LlamaFunctionCallingHandler()
        
        # Invalid JSON
        response = b"not valid json"
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Invalid JSON response from DJL"):
            handler.transform_output(response)
    
    def test_transform_output_missing_choices(self):
        """Test transform_output with missing choices field."""
        from src.content_handler import LlamaFunctionCallingHandler
        
        handler = LlamaFunctionCallingHandler()
        
        # Missing choices field
        response = json.dumps({"wrong_field": "value"}).encode("utf-8")
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="No choices in response"):
            handler.transform_output(response)
    
    def test_transform_output_empty_choices(self):
        """Test transform_output with empty choices array."""
        from src.content_handler import LlamaFunctionCallingHandler
        
        handler = LlamaFunctionCallingHandler()
        
        # Empty choices
        response = json.dumps({"choices": []}).encode("utf-8")
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="No choices in response"):
            handler.transform_output(response)
    
    def test_transform_output_malformed_tool_call(self):
        """Test transform_output with malformed tool call JSON."""
        from src.content_handler import LlamaFunctionCallingHandler
        
        handler = LlamaFunctionCallingHandler()
        
        # Malformed tool call (invalid JSON in arguments)
        response = json.dumps({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "yahoo_stock_price",
                            "arguments": "{invalid json}"
                        }
                    }]
                }
            }]
        }).encode("utf-8")
        
        result = handler.transform_output(response)
        
        # Should skip malformed tool call and return empty
        assert isinstance(result, AIMessage)
        assert result.tool_calls == []
    
    def test_transform_output_empty_content(self):
        """Test transform_output with empty content."""
        from src.content_handler import LlamaFunctionCallingHandler
        
        handler = LlamaFunctionCallingHandler()
        
        # Empty content
        response = json.dumps({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": ""
                }
            }]
        }).encode("utf-8")
        
        result = handler.transform_output(response)
        
        # Should return AIMessage with empty content
        assert isinstance(result, AIMessage)
        assert result.content == ""
        assert result.tool_calls == []
    
    def test_transform_output_logging_on_success(self, caplog):
        """Test that successful tool call detection is logged."""
        import logging
        from src.content_handler import LlamaFunctionCallingHandler
        
        caplog.set_level(logging.INFO)
        handler = LlamaFunctionCallingHandler()
        
        response = json.dumps({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "yahoo_stock_price",
                            "arguments": '{"ticker": "AAPL"}'
                        }
                    }]
                }
            }]
        }).encode("utf-8")
        
        result = handler.transform_output(response)
        
        # Verify tool call was detected
        assert len(result.tool_calls) == 1
        
        # Verify logging occurred
        assert "Detected 1 tool call(s)" in caplog.text
        assert "yahoo_stock_price" in caplog.text
    
    def test_transform_output_diagnostic_info_on_json_error(self, caplog):
        """Test that JSON decode errors provide comprehensive diagnostic information."""
        import logging
        from src.content_handler import LlamaFunctionCallingHandler
        
        caplog.set_level(logging.ERROR)
        handler = LlamaFunctionCallingHandler()
        
        # Invalid JSON
        response = b"not valid json at all"
        
        with pytest.raises(ValueError, match="Invalid JSON response from DJL"):
            handler.transform_output(response)
        
        # Check diagnostic information in logs
        log_text = caplog.text
        assert "JSONDecodeError" in log_text or "Failed to parse" in log_text
    
    def test_transform_output_diagnostic_info_on_invalid_format(self, caplog):
        """Test diagnostic information for invalid response format."""
        import logging
        from src.content_handler import LlamaFunctionCallingHandler
        
        caplog.set_level(logging.ERROR)
        handler = LlamaFunctionCallingHandler()
        
        # Invalid format (missing choices)
        response = json.dumps({"wrong": "format"}).encode("utf-8")
        
        with pytest.raises(ValueError, match="No choices in response"):
            handler.transform_output(response)
        
        # Check diagnostic information
        log_text = caplog.text
        assert "InvalidResponseFormat" in log_text or "Invalid response format" in log_text
    
    def test_transform_output_handles_malformed_without_crashing(self):
        """Test that malformed responses are handled gracefully without crashing."""
        from src.content_handler import LlamaFunctionCallingHandler
        
        handler = LlamaFunctionCallingHandler()
        
        # Various malformed responses that should not crash
        malformed_responses = [
            # Malformed tool call - should skip and return empty
            json.dumps({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "function": {
                                "name": "test",
                                "arguments": "{bad json}"
                            }
                        }]
                    }
                }]
            }).encode("utf-8"),
            # Empty string - should return empty AIMessage
            json.dumps({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": ""
                    }
                }]
            }).encode("utf-8"),
            # Regular text - should return as content
            json.dumps({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Just a regular response"
                    }
                }]
            }).encode("utf-8"),
        ]
        
        for response in malformed_responses:
            result = handler.transform_output(response)
            # Should always return an AIMessage, never crash
            assert isinstance(result, AIMessage)
    
    def test_transform_output_fallback_on_tool_call_error(self, caplog):
        """Test that tool call parsing errors are logged and skipped."""
        import logging
        from src.content_handler import LlamaFunctionCallingHandler
        
        caplog.set_level(logging.WARNING)
        handler = LlamaFunctionCallingHandler()
        
        # Malformed tool call that will fail parsing
        response = json.dumps({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Some text",
                    "tool_calls": [{
                        "id": "call_123",
                        "function": {
                            "name": "test_tool",
                            "arguments": "{invalid: json}"
                        }
                    }]
                }
            }]
        }).encode("utf-8")
        
        result = handler.transform_output(response)
        
        # Should skip malformed tool call
        assert isinstance(result, AIMessage)
        assert result.tool_calls == []
        assert result.content == "Some text"
        
        # Should have logged the parsing error
        assert "Failed to parse tool call arguments" in caplog.text
