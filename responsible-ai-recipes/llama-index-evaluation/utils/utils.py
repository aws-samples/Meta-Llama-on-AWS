# Note: remove meta llama 2 as this is soon deprecated
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.base.llms.generic_utils import (
    prompt_to_messages,
)
from llama_index.llms.anthropic.utils import messages_to_anthropic_messages
# Add specific prompt template for Meta Llama 3
from utils.llama_utils import (
    completion_to_prompt as completion_to_llama_prompt,
)
from utils.llama_utils import (
    messages_to_prompt as messages_to_llama_prompt,
)
# Mistral will be using previous version per prompt template
from llama_index.llms.bedrock.llama_utils import (
    completion_to_prompt as completion_to_mistral_prompt,
)
from llama_index.llms.bedrock.llama_utils import (
    messages_to_prompt as messages_to_mistral_prompt,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

HUMAN_PREFIX = "\n\nHuman:"
ASSISTANT_PREFIX = "\n\nAssistant:"

# Values taken from 
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html
COMPLETION_MODELS = {
    "amazon.titan-tg1-large": 8000,
    "amazon.titan-text-express-v1": 8000,
    "ai21.j2-grande-instruct": 8000,
    "ai21.j2-jumbo-instruct": 8000,
    "ai21.j2-mid": 8000,
    "ai21.j2-mid-v1": 8000,
    "ai21.j2-ultra": 8000,
    "ai21.j2-ultra-v1": 8000,
    "cohere.command-text-v14": 4096,
}

# Anthropic models require prompt to start with "Human:" and
# end with "Assistant:"
# Add meta llama 3.1 support with context length
CHAT_ONLY_MODELS = {
    "anthropic.claude-instant-v1": 100000,
    "anthropic.claude-v1": 100000,
    "anthropic.claude-v2": 100000,
    "anthropic.claude-v2:1": 200000,
    "anthropic.claude-3-sonnet-20240229-v1:0": 200000,
    "anthropic.claude-3-haiku-20240307-v1:0": 200000,
    "anthropic.claude-3-opus-20240229-v1:0": 200000,
    "anthropic.claude-3-5-sonnet-20240620-v1:0": 200000,
    "anthropic.claude-3-5-sonnet-20241022-v2:0": 200000,
    "cohere.command-r-plus-v1:0": 128000,
    "meta.llama3-8b-instruct-v1:0": 8192,
    "meta.llama3-70b-instruct-v1:0": 8192,
    "meta.llama3-1-8b-instruct-v1:0": 128000,
    "meta.llama3-1-70b-instruct-v1:0": 128000,
    "meta.llama3-1-405b-instruct-v1:0": 128000,
    "mistral.mistral-7b-instruct-v0:2": 32000,
    "mistral.mixtral-8x7b-instruct-v0:1": 32000,
    "mistral.mistral-large-2402-v1:0": 32000,
    "mistral.mistral-large-2407-v1:0": 32000,
    "ai21.jamba-1-5-mini-v1:0": 256000,
    "ai21.jamba-1-5-large-v1:0": 256000,
}
BEDROCK_FOUNDATION_LLMS = {**COMPLETION_MODELS, **CHAT_ONLY_MODELS}

# Only the following models support streaming as
# per result of Bedrock.Client.list_foundation_models
# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock/client/list_foundation_models.html
# Add Meta llama 3 and 3.1 to streaming
STREAMING_MODELS = {
    "amazon.titan-tg1-large",
    "amazon.titan-text-express-v1",
    "anthropic.claude-instant-v1",
    "anthropic.claude-v1",
    "anthropic.claude-v2",
    "anthropic.claude-v2:1",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "meta.llama3-8b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "meta.llama3-1-8b-instruct-v1:0",
    "meta.llama3-1-70b-instruct-v1:0",
    "meta.llama3-1-405b-instruct-v1:0",
    "mistral.mistral-7b-instruct-v0:2",
    "mistral.mixtral-8x7b-instruct-v0:1",
    "mistral.mistral-large-2402-v1:0",
}


class Provider(ABC):
    @property
    @abstractmethod
    def max_tokens_key(self) -> str:
        ...

    @abstractmethod
    def get_text_from_response(self, response: dict) -> str:
        ...

    def get_text_from_stream_response(self, response: dict) -> str:
        return self.get_text_from_response(response)

    def get_request_body(self, prompt: str, inference_parameters: dict) -> dict:
        return {"prompt": prompt, **inference_parameters}

    messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None
    completion_to_prompt: Optional[Callable[[str], str]] = None


class AmazonProvider(Provider):
    max_tokens_key = "maxTokenCount"

    def get_text_from_response(self, response: dict) -> str:
        return response["results"][0]["outputText"]

    def get_text_from_stream_response(self, response: dict) -> str:
        return response["outputText"]

    def get_request_body(self, prompt: str, inference_parameters: dict) -> dict:
        return {
            "inputText": prompt,
            "textGenerationConfig": {**inference_parameters},
        }


class Ai21Provider(Provider):
    max_tokens_key = "maxTokens"

    def get_text_from_response(self, response: dict) -> str:
        return response["completions"][0]["data"]["text"]


def completion_to_anthopic_prompt(completion: str) -> str:
    messages, _ = messages_to_anthropic_messages(prompt_to_messages(completion))
    return messages


def _messages_to_anthropic_messages(messages: Sequence[ChatMessage]) -> List[dict]:
    messages, system_prompt = messages_to_anthropic_messages(messages)
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}, *messages]
    return messages


class AnthropicProvider(Provider):
    max_tokens_key = "max_tokens"

    def __init__(self) -> None:
        self.messages_to_prompt = _messages_to_anthropic_messages
        self.completion_to_prompt = completion_to_anthopic_prompt

    def get_text_from_stream_response(self, response: dict) -> str:
        if response["type"] == "content_block_delta":
            return response["delta"]["text"]
        else:
            return ""

    def get_text_from_response(self, response: dict) -> str:
        if response["content"]:
            return response["content"][0]["text"]
        return ""

    def get_request_body(self, prompt: Sequence[Dict], inference_parameters: dict):
        if len(prompt) > 0 and prompt[0]["role"] == "system":
            system_message = prompt[0]["content"]
            prompt = prompt[1:]

            if (
                "system" in inference_parameters
                and inference_parameters["system"] is not None
            ):
                inference_parameters["system"] += system_message
            else:
                inference_parameters["system"] = system_message

        return {
            "messages": prompt,
            "anthropic_version": inference_parameters.get(
                "anthropic_version", "bedrock-2023-05-31"
            ),  # Required by AWS.
            **inference_parameters,
        }


class CohereProvider(Provider):
    max_tokens_key = "max_tokens"

    def get_text_from_response(self, response: dict) -> str:
        return response["generations"][0]["text"]


class MetaProvider(Provider):
    max_tokens_key = "max_gen_len"

    def __init__(self) -> None:
        self.messages_to_prompt = messages_to_llama_prompt
        self.completion_to_prompt = completion_to_llama_prompt

    def get_text_from_response(self, response: dict) -> str:
        return response["generation"]


class MistralProvider(Provider):
    max_tokens_key = "max_tokens"

    def __init__(self) -> None:
        self.messages_to_prompt = messages_to_mistral_prompt
        self.completion_to_prompt = completion_to_mistral_prompt

    def get_text_from_response(self, response: dict) -> str:
        return response["outputs"][0]["text"]


PROVIDERS = {
    "amazon": AmazonProvider(),
    "ai21": Ai21Provider(),
    "anthropic": AnthropicProvider(),
    "cohere": CohereProvider(),
    "meta": MetaProvider(),
    "mistral": MistralProvider(),
}


def get_provider(model: str) -> Provider:
    provider_name = model.split(".")[0]
    if provider_name not in PROVIDERS:
        raise ValueError(f"Provider {provider_name} \
            for model {model} is not supported")
    return PROVIDERS[provider_name]


logger = logging.getLogger(__name__)


def _create_retry_decorator(client: Any, max_retries: int) -> Callable[[Any], Any]:
    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    try:
        import boto3  # noqa
    except ImportError as e:
        raise ImportError(
            "You must install the `boto3` package to use Bedrock."
            "Please `pip install boto3`"
        ) from e

    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type(client.exceptions.ThrottlingException)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def completion_with_retry(
    client: Any,
    model: str,
    request_body: str,
    max_retries: int,
    stream: bool = False,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the completion call."""
    retry_decorator = _create_retry_decorator(
        client=client,
        max_retries=max_retries
    )

    @retry_decorator
    def _completion_with_retry(**kwargs: Any) -> Any:
        if stream:
            return client.invoke_model_with_response_stream(
                modelId=model, body=request_body
            )
        return client.invoke_model(modelId=model, body=request_body)

    return _completion_with_retry(**kwargs)