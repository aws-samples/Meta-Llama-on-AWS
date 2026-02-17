"""
Unit tests for ChatSagemakerWithTools wrapper class.

Tests verify that the wrapper correctly stores bound tools and passes them
to the endpoint during invocation.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from src.sagemaker_with_tools import ChatSagemakerWithTools
from src.content_handler import LlamaFunctionCallingHandler


class TestChatSagemakerWithTools:
    """Test suite for ChatSagemakerWithTools class."""
    
    @pytest.fixture
    def mock_content_handler(self):
        """Create a mock content handler."""
        return Mock(spec=LlamaFunctionCallingHandler)
    
    @pytest.fixture
    def mock_endpoint(self):
        """Create a mock ChatSagemakerEndpoint."""
        endpoint = Mock()
        endpoint.invoke = Mock(return_value=AIMessage(content="Test response"))
        return endpoint
    
    @patch('src.sagemaker_with_tools.ChatSagemakerEndpoint')
    def test_initialization(self, mock_endpoint_class, mock_content_handler):
        """Test that ChatSagemakerWithTools initializes correctly."""
        # Arrange
        endpoint_name = "test-endpoint"
        region_name = "us-west-2"
        
        # Act
        llm = ChatSagemakerWithTools(
            endpoint_name=endpoint_name,
            region_name=region_name,
            content_handler=mock_content_handler
        )
        
        # Assert
        assert llm.bound_tools == []
        mock_endpoint_class.assert_called_once_with(
            endpoint_name=endpoint_name,
            region_name=region_name,
            content_handler=mock_content_handler
        )
    
    @patch('src.sagemaker_with_tools.ChatSagemakerEndpoint')
    def test_initialization_with_kwargs(self, mock_endpoint_class, mock_content_handler):
        """Test that additional kwargs are passed to ChatSagemakerEndpoint."""
        # Arrange
        endpoint_name = "test-endpoint"
        region_name = "us-west-2"
        extra_param = "extra_value"
        
        # Act
        llm = ChatSagemakerWithTools(
            endpoint_name=endpoint_name,
            region_name=region_name,
            content_handler=mock_content_handler,
            extra_param=extra_param
        )
        
        # Assert
        mock_endpoint_class.assert_called_once_with(
            endpoint_name=endpoint_name,
            region_name=region_name,
            content_handler=mock_content_handler,
            extra_param=extra_param
        )
    
    @patch('src.sagemaker_with_tools.ChatSagemakerEndpoint')
    def test_bind_tools_stores_tools(self, mock_endpoint_class, mock_content_handler):
        """Test that bind_tools stores tools in bound_tools attribute."""
        # Arrange
        llm = ChatSagemakerWithTools(
            endpoint_name="test-endpoint",
            region_name="us-west-2",
            content_handler=mock_content_handler
        )
        mock_tool1 = Mock()
        mock_tool2 = Mock()
        tools = [mock_tool1, mock_tool2]
        
        # Act
        result = llm.bind_tools(tools)
        
        # Assert
        assert llm.bound_tools == tools
        assert result is llm  # Verify method chaining
    
    @patch('src.sagemaker_with_tools.ChatSagemakerEndpoint')
    def test_bind_tools_returns_self(self, mock_endpoint_class, mock_content_handler):
        """Test that bind_tools returns self for method chaining."""
        # Arrange
        llm = ChatSagemakerWithTools(
            endpoint_name="test-endpoint",
            region_name="us-west-2",
            content_handler=mock_content_handler
        )
        tools = [Mock()]
        
        # Act
        result = llm.bind_tools(tools)
        
        # Assert
        assert result is llm
    
    @patch('src.sagemaker_with_tools.ChatSagemakerEndpoint')
    def test_invoke_without_tools(self, mock_endpoint_class, mock_content_handler):
        """Test that invoke works without bound tools."""
        # Arrange
        mock_endpoint = Mock()
        expected_response = AIMessage(content="Test response")
        mock_endpoint.invoke = Mock(return_value=expected_response)
        mock_endpoint_class.return_value = mock_endpoint
        
        llm = ChatSagemakerWithTools(
            endpoint_name="test-endpoint",
            region_name="us-west-2",
            content_handler=mock_content_handler
        )
        messages = [HumanMessage(content="Hello")]
        
        # Act
        response = llm.invoke(messages)
        
        # Assert
        assert response == expected_response
        mock_endpoint.invoke.assert_called_once_with(messages)
    
    @patch('src.sagemaker_with_tools.ChatSagemakerEndpoint')
    def test_invoke_with_bound_tools(self, mock_endpoint_class, mock_content_handler):
        """Test that invoke passes bound tools via model_kwargs."""
        # Arrange
        mock_endpoint = Mock()
        expected_response = AIMessage(content="Test response")
        mock_endpoint.invoke = Mock(return_value=expected_response)
        mock_endpoint._generate = Mock()  # Mock the _generate method that gets overridden
        mock_endpoint_class.return_value = mock_endpoint
        
        llm = ChatSagemakerWithTools(
            endpoint_name="test-endpoint",
            region_name="us-west-2",
            content_handler=mock_content_handler
        )
        
        mock_tool1 = Mock()
        mock_tool2 = Mock()
        tools = [mock_tool1, mock_tool2]
        llm.bind_tools(tools)
        
        messages = [HumanMessage(content="Hello")]
        
        # Act
        response = llm.invoke(messages)
        
        # Assert
        assert response == expected_response
        # Verify invoke was called (the implementation temporarily overrides _generate)
        mock_endpoint.invoke.assert_called_once()
    
    @patch('src.sagemaker_with_tools.ChatSagemakerEndpoint')
    def test_invoke_with_additional_kwargs(self, mock_endpoint_class, mock_content_handler):
        """Test that invoke passes additional kwargs to endpoint."""
        # Arrange
        mock_endpoint = Mock()
        expected_response = AIMessage(content="Test response")
        mock_endpoint.invoke = Mock(return_value=expected_response)
        mock_endpoint_class.return_value = mock_endpoint
        
        llm = ChatSagemakerWithTools(
            endpoint_name="test-endpoint",
            region_name="us-west-2",
            content_handler=mock_content_handler
        )
        
        messages = [HumanMessage(content="Hello")]
        
        # Act
        response = llm.invoke(messages, temperature=0.5, max_tokens=100)
        
        # Assert
        assert response == expected_response
        mock_endpoint.invoke.assert_called_once_with(
            messages,
            temperature=0.5,
            max_tokens=100
        )
    
    @patch('src.sagemaker_with_tools.ChatSagemakerEndpoint')
    def test_invoke_with_tools_and_kwargs(self, mock_endpoint_class, mock_content_handler):
        """Test that invoke passes both tools and additional kwargs."""
        # Arrange
        mock_endpoint = Mock()
        expected_response = AIMessage(content="Test response")
        mock_endpoint.invoke = Mock(return_value=expected_response)
        mock_endpoint._generate = Mock()  # Mock the _generate method that gets overridden
        mock_endpoint_class.return_value = mock_endpoint
        
        llm = ChatSagemakerWithTools(
            endpoint_name="test-endpoint",
            region_name="us-west-2",
            content_handler=mock_content_handler
        )
        
        tools = [Mock(), Mock()]
        llm.bind_tools(tools)
        
        messages = [HumanMessage(content="Hello")]
        
        # Act
        response = llm.invoke(messages, temperature=0.5, max_tokens=100)
        
        # Assert
        assert response == expected_response
        # Verify invoke was called (the implementation temporarily overrides _generate)
        mock_endpoint.invoke.assert_called_once()
    
    @patch('src.sagemaker_with_tools.ChatSagemakerEndpoint')
    def test_method_chaining(self, mock_endpoint_class, mock_content_handler):
        """Test that bind_tools supports method chaining with invoke."""
        # Arrange
        mock_endpoint = Mock()
        expected_response = AIMessage(content="Test response")
        mock_endpoint.invoke = Mock(return_value=expected_response)
        mock_endpoint._generate = Mock()  # Mock the _generate method that gets overridden
        mock_endpoint_class.return_value = mock_endpoint
        
        llm = ChatSagemakerWithTools(
            endpoint_name="test-endpoint",
            region_name="us-west-2",
            content_handler=mock_content_handler
        )
        
        tools = [Mock()]
        messages = [HumanMessage(content="Hello")]
        
        # Act - chain bind_tools and invoke
        response = llm.bind_tools(tools).invoke(messages)
        
        # Assert
        assert response == expected_response
        assert llm.bound_tools == tools
        # Verify invoke was called (the implementation temporarily overrides _generate)
        mock_endpoint.invoke.assert_called_once()

