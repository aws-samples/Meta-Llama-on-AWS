"""AWS Bedrock LLM client for PoC agents with Llama support."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterator, List

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

from src.orchestration.observability import emit_event, wrap_payload

load_dotenv()

logger = logging.getLogger(__name__)

_CLIENT: boto3.client | None = None
_REGION = os.getenv("AWS_REGION", "us-east-2")
_DEFAULT_TIMEOUT = float(os.getenv("BEDROCK_TIMEOUT", "60"))


class BedrockClientManager:
    """Manages AWS Bedrock client lifecycle with proper resource cleanup."""

    def __init__(self) -> None:
        self._client: boto3.client | None = None

    def get_client(self) -> boto3.client:
        """Get or create the Bedrock runtime client with proper configuration."""
        if self._client is None:
            # Set up boto3 client configuration
            client_config = Config(
                retries={"max_attempts": 3, "mode": "adaptive"},
                read_timeout=_DEFAULT_TIMEOUT,
                connect_timeout=10,
                region_name=_REGION,
            )

            # Build client parameters
            client_params = {
                "service_name": "bedrock-runtime",
                "region_name": _REGION,
                "config": client_config,
            }

            # Use credentials from environment variables (loaded via .env)
            aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_session_token = os.getenv("AWS_SESSION_TOKEN")

            if aws_access_key_id and aws_secret_access_key:
                client_params["aws_access_key_id"] = aws_access_key_id
                client_params["aws_secret_access_key"] = aws_secret_access_key
                if aws_session_token:
                    client_params["aws_session_token"] = aws_session_token

            try:
                self._client = boto3.client(**client_params)
                logger.info(f"Initialized Bedrock client for region: {_REGION}")
            except Exception as e:
                logger.error(f"Failed to initialize AWS Bedrock client: {e}")
                raise ConnectionError(f"Failed to initialize AWS Bedrock client: {e}")

        return self._client

    def close(self) -> None:
        """Close the Bedrock client and clean up resources."""
        if self._client is not None:
            # Boto3 clients don't need explicit closing, but we can clear the reference
            self._client = None
            logger.info("Bedrock client closed")


_CLIENT_MANAGER = BedrockClientManager()


def bedrock_client() -> boto3.client:
    """Get the shared Bedrock runtime client instance."""
    return _CLIENT_MANAGER.get_client()


def validate_api_key_at_startup() -> None:
    """Validate AWS Bedrock credentials and access at application startup."""
    try:
        # Test Bedrock access by trying to list available models
        bedrock_control = boto3.client(
            "bedrock",
            region_name=_REGION,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        )

        # Attempt to list foundation models to verify access
        response = bedrock_control.list_foundation_models()
        model_count = len(response.get("modelSummaries", []))

        if model_count == 0:
            raise ValueError("No foundation models found - check Bedrock permissions")

        logger.info(
            f"✅ AWS Bedrock validation successful - {model_count} models available"
        )

        # Also test the runtime client
        runtime_client = bedrock_client()
        logger.info("✅ Bedrock runtime client initialized successfully")

    except NoCredentialsError:
        raise ValueError(
            "AWS credentials not found. Please ensure AWS_ACCESS_KEY_ID, "
            "AWS_SECRET_ACCESS_KEY, and optionally AWS_SESSION_TOKEN are set in your .env file"
        )
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "UnauthorizedOperation":
            raise ValueError(
                "AWS credentials lack Bedrock permissions. Please ensure your role has "
                "bedrock:ListFoundationModels and bedrock-runtime:InvokeModel permissions"
            )
        else:
            raise ValueError(f"AWS Bedrock validation failed: {error_code} - {e}")
    except Exception as e:
        raise ValueError(f"AWS Bedrock validation failed: {e}")


def close_bedrock_client() -> None:
    """Close the Bedrock client and clean up resources."""
    global _CLIENT_MANAGER
    _CLIENT_MANAGER.close()


def _prepare_messages(
    messages: List[Dict[str, Any]], system: str
) -> List[Dict[str, str]]:
    """Prepare chat messages for Bedrock, injecting system message if provided."""
    chat_messages = []

    # Add system message first if provided
    if system:
        chat_messages.append({"role": "system", "content": system})

    # Add user messages
    for message in messages:
        chat_messages.append(
            {
                "role": str(message.get("role", "user")),
                "content": str(message.get("content", "")),
            }
        )

    return chat_messages


def _format_llama_prompt(messages: List[Dict[str, str]]) -> str:
    """Format messages for Llama with proper special tokens."""
    formatted_prompt = "<|begin_of_text|>"

    for message in messages:
        role = message["role"]
        content = message["content"]
        formatted_prompt += (
            f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"
        )

    formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return formatted_prompt


def converse_claude(
    messages: List[Dict[str, Any]],
    system: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call AWS Bedrock with Llama model and return text content."""

    client = bedrock_client()
    chat_messages = _prepare_messages(messages, system)

    # Format messages for Llama using special tokens
    prompt = _format_llama_prompt(chat_messages)

    # Prepare payload for Bedrock invoke_model
    payload = {
        "prompt": prompt,
        "max_gen_len": max(1, max_tokens),
        "temperature": temperature,
        "top_p": 0.9,
    }

    emit_event(
        "bedrock_client",
        "llm_request",
        wrap_payload(
            model=model_id,
            region=_REGION,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        ),
    )


    try:
        response = client.invoke_model(
            modelId=model_id, body=json.dumps(payload), contentType="application/json"
        )

        # Parse response
        result = json.loads(response["body"].read())

        content = result.get("generation", "")

        if not content:
            logger.warning("Empty response from Bedrock model")
            return "{}"  # Return empty JSON object as fallback

        emit_event(
            "bedrock_client",
            "llm_success",
            wrap_payload(model=model_id, response_length=len(content), stream=False),
        )

        return content.strip()

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_message = e.response.get("Error", {}).get("Message", str(e))

        emit_event(
            "bedrock_client",
            "llm_error",
            wrap_payload(
                model=model_id, error_code=error_code, error_message=error_message
            ),
        )

        logger.error(f"Bedrock API error ({error_code}): {error_message}")
        raise
    except Exception as e:
        emit_event(
            "bedrock_client",
            "llm_error",
            wrap_payload(
                model=model_id, error_type=type(e).__name__, error_message=str(e)
            ),
        )

        logger.error(f"Bedrock client error: {e}")
        raise


def converse_claude_stream(
    messages: List[Dict[str, Any]],
    system: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
) -> Iterator[str]:
    """Yield streaming content chunks from AWS Bedrock with Llama model."""

    client = bedrock_client()
    chat_messages = _prepare_messages(messages, system)

    # Format messages for Llama using special tokens
    prompt = _format_llama_prompt(chat_messages)

    # Prepare payload for Bedrock invoke_model_with_response_stream
    payload = {
        "prompt": prompt,
        "max_gen_len": max(1, max_tokens),
        "temperature": temperature,
        "top_p": 0.9,
    }

    emit_event(
        "bedrock_client",
        "llm_stream_request",
        wrap_payload(
            model=model_id,
            region=_REGION,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ),
    )

    try:
        response = client.invoke_model_with_response_stream(
            modelId=model_id, body=json.dumps(payload), contentType="application/json"
        )

        # Process streaming response
        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])

            if "generation" in chunk:
                generation_text = chunk["generation"]
                if generation_text:
                    yield generation_text

        emit_event("bedrock_client", "llm_stream_success", wrap_payload(model=model_id))

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_message = e.response.get("Error", {}).get("Message", str(e))

        emit_event(
            "bedrock_client",
            "llm_stream_error",
            wrap_payload(
                model=model_id, error_code=error_code, error_message=error_message
            ),
        )

        logger.error(f"Bedrock streaming API error ({error_code}): {error_message}")
        raise
    except Exception as e:
        emit_event(
            "bedrock_client",
            "llm_stream_error",
            wrap_payload(
                model=model_id, error_type=type(e).__name__, error_message=str(e)
            ),
        )

        logger.error(f"Bedrock streaming error: {e}")
        raise
