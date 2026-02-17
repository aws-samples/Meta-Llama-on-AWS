"""
Wrapper around ChatSagemakerEndpoint that supports bind_tools.

This module provides the ChatSagemakerWithTools class that mimics the interface
of ChatOpenAI to minimize changes to existing agent code when migrating to SageMaker.
"""

from typing import List, Any, Dict
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_aws.chat_models.sagemaker_endpoint import ChatSagemakerEndpoint

from .content_handler import LlamaFunctionCallingHandler


class ChatSagemakerWithTools:
    """
    Wrapper around ChatSagemakerEndpoint that supports bind_tools.
    
    This class mimics the interface of ChatOpenAI to minimize changes to existing
    agent code. It stores bound tools and passes them to the content handler via
    model_kwargs during invocation.
    
    Attributes:
        endpoint: ChatSagemakerEndpoint instance for SageMaker inference
        bound_tools: List of tools bound to this LLM instance
    
    Example:
        >>> handler = LlamaFunctionCallingHandler()
        >>> llm = ChatSagemakerWithTools(
        ...     endpoint_name="my-llama-endpoint",
        ...     region_name="us-west-2",
        ...     content_handler=handler
        ... )
        >>> llm_with_tools = llm.bind_tools([tool1, tool2])
        >>> response = llm_with_tools.invoke([HumanMessage(content="Hello")])
    """
    
    def __init__(
        self,
        endpoint_name: str,
        region_name: str,
        content_handler: LlamaFunctionCallingHandler,
        **kwargs
    ):
        """
        Initialize ChatSagemakerWithTools.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            region_name: AWS region where the endpoint is deployed
            content_handler: LlamaFunctionCallingHandler instance for message transformation
            **kwargs: Additional arguments passed to ChatSagemakerEndpoint
        """
        # Store ChatSagemakerEndpoint instance
        self.endpoint = ChatSagemakerEndpoint(
            endpoint_name=endpoint_name,
            region_name=region_name,
            content_handler=content_handler,
            **kwargs
        )
        
        # Initialize empty bound_tools list
        self.bound_tools = []
    
    def bind_tools(self, tools: List) -> "ChatSagemakerWithTools":
        """
        Bind tools to the LLM.
        
        Stores tool schemas that will be injected into prompts by the content handler.
        Returns self to support method chaining.
        
        Args:
            tools: List of LangChain tool objects to bind
        
        Returns:
            Self for method chaining
        
        Example:
            >>> llm = ChatSagemakerWithTools(...)
            >>> llm_with_tools = llm.bind_tools([tool1, tool2])
            >>> # Can now invoke with tools available
            >>> response = llm_with_tools.invoke(messages)
        """
        # Store tools in bound_tools attribute
        self.bound_tools = tools
        
        # Return self for chaining
        return self
    
    def invoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        """
        Invoke the model with messages.
        
        Passes bound tools to the content handler via model_kwargs so they can be
        included in the prompt. Overrides _generate to bypass langchain_aws's message
        conversion which doesn't support ToolMessage.
        
        Args:
            messages: List of LangChain message objects
            **kwargs: Additional keyword arguments passed to endpoint
        
        Returns:
            AIMessage result from the model
        
        Example:
            >>> llm = ChatSagemakerWithTools(...).bind_tools([tool1, tool2])
            >>> messages = [HumanMessage(content="What's the weather?")]
            >>> response = llm.invoke(messages)
            >>> # response may contain tool_calls if model decided to use tools
        """
        # Pass bound_tools to endpoint via model_kwargs
        # This allows the content handler to access tools during transform_input
        model_kwargs = kwargs.copy()
        if self.bound_tools:
            model_kwargs["tools"] = self.bound_tools
        
        # Override _generate to bypass langchain_aws message conversion
        # This is necessary because langchain_aws doesn't support ToolMessage
        original_generate = self.endpoint._generate
        
        def bypass_generate(messages_input: List[BaseMessage], stop=None, **gen_kwargs):
            # Merge model_kwargs with gen_kwargs (gen_kwargs takes precedence)
            merged_kwargs = {**model_kwargs, **gen_kwargs}
            
            # Ensure max_tokens is used correctly (not max_new_tokens)
            if "max_tokens" in merged_kwargs:
                merged_kwargs["max_new_tokens"] = merged_kwargs.pop("max_tokens")
            
            # Call content handler directly
            body = self.endpoint.content_handler.transform_input(messages_input, merged_kwargs)
            
            # Invoke endpoint
            response = self.endpoint.client.invoke_endpoint(
                EndpointName=self.endpoint.endpoint_name,
                Body=body,
                ContentType=self.endpoint.content_handler.content_type,
                Accept=self.endpoint.content_handler.accepts,
            )
            
            # Transform output
            ai_message = self.endpoint.content_handler.transform_output(response["Body"])
            
            # Return ChatResult
            generation = ChatGeneration(message=ai_message)
            return ChatResult(generations=[generation])
        
        try:
            # Temporarily replace the _generate method
            self.endpoint._generate = bypass_generate
            
            # Call endpoint.invoke with messages and kwargs
            return self.endpoint.invoke(messages, **kwargs)
        finally:
            # Restore original method
            self.endpoint._generate = original_generate
