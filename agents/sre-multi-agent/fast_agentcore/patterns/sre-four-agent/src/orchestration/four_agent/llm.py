"""LLM infrastructure shared by four-agent implementations with support for multiple providers."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    TYPE_CHECKING,
)

from ..observability import emit_event, wrap_payload
from .settings import (
    DEFAULT_LLM_PROVIDER,
    get_default_max_tokens,
    get_default_model,
    get_default_temperature,
    MAX_JSON_RESPONSE_SIZE,
)

if TYPE_CHECKING:
    from .schema import AgentMessage
    from .state import IncidentState

logger = logging.getLogger(__name__)


def _is_retryable_error(exception: Exception) -> bool:
    """Determine if an error should be retried based on its type and properties.

    Args:
        exception: The exception to evaluate

    Returns:
        True if the error is retryable, False otherwise
    """
    # Import here to avoid circular dependencies
    try:
        import httpx

        # Retryable HTTP errors
        if isinstance(
            exception, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadTimeout)
        ):
            return True

        # Retryable HTTP status errors (rate limits, server errors)
        if isinstance(exception, httpx.HTTPStatusError):
            if exception.response.status_code in (429, 500, 502, 503, 504):
                return True

    except ImportError:
        pass  # httpx not available, skip HTTP-specific checks

    # Generic network/temporary errors
    error_message = str(exception).lower()
    retryable_patterns = [
        "timeout",
        "connection",
        "network",
        "temporary",
        "rate limit",
        "throttle",
        "overload",
        "busy",
    ]

    if any(pattern in error_message for pattern in retryable_patterns):
        return True

    # Non-retryable errors
    non_retryable_types = (
        KeyboardInterrupt,
        SystemExit,
        MemoryError,
        ValueError,
        TypeError,
        AttributeError,  # Logic/validation errors
    )

    if isinstance(exception, non_retryable_types):
        return False

    # Default: retry generic exceptions (like LLM service errors)
    return True


def _calculate_retry_delay(
    attempt: int, base_delay: float = 1.0, max_delay: float = 60.0
) -> float:
    """Calculate exponential backoff delay with jitter.

    Args:
        attempt: The attempt number (1-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Delay in seconds with exponential backoff and jitter
    """
    # Exponential backoff: 2^(attempt-1) * base_delay
    exponential_delay = base_delay * (2 ** (attempt - 1))

    # Cap at maximum delay
    capped_delay = min(exponential_delay, max_delay)

    # Add jitter: random between 0.5x and 1.5x the calculated delay
    jitter_factor = 0.5 + random.random()  # Random between 0.5 and 1.5
    final_delay = capped_delay * jitter_factor

    return final_delay


class LLMResponseFormatError(RuntimeError):
    """Raised when an LLM response cannot be parsed into the expected schema."""


@dataclass
class LLMRequest:
    """Container describing an outbound LLM request."""

    system_prompt: str
    messages: Sequence[Mapping[str, str]]
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass
class LLMResult:
    """Minimal response wrapper returned by an :class:`LLMRunner`."""

    text: str
    raw: Optional[Any] = None
    usage: Optional[Mapping[str, Any]] = None


class LLMRunner(Protocol):
    """Protocol implemented by objects capable of executing LLM requests."""

    async def run(
        self,
        request: LLMRequest,
        *,
        stream: Optional[Callable[[str], None]] = None,
    ) -> LLMResult: ...


@dataclass
class LLMProviderConfig:
    """Configuration for LLM provider-specific operations."""

    event_prefix: str  # e.g., "bedrock_runner"
    stream_function: Callable  # e.g., converse_claude_stream
    chat_function: Callable  # e.g., converse_claude


class BaseChatRunner(LLMRunner):
    """Base class for chat completion runners with common functionality."""

    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        provider_config: LLMProviderConfig,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._provider = provider_config

    async def run(
        self,
        request: LLMRequest,
        *,
        stream: Optional[Callable[[str], None]] = None,
    ) -> LLMResult:
        """Execute LLM request with provider-specific implementation."""
        model = request.model or self._model
        max_tokens = request.max_tokens or self._max_tokens
        temperature = (
            request.temperature
            if request.temperature is not None
            else self._temperature
        )

        def _execute() -> LLMResult:
            messages = list(request.messages)
            system_prompt = request.system_prompt

            # Emit invocation event
            emit_event(
                self._provider.event_prefix,
                "llm_invocation",
                wrap_payload(
                    model=model, stream=request.stream, messages=len(messages)
                ),
            )

            # Handle streaming requests
            if request.stream and stream is not None:
                return self._handle_streaming_request(
                    messages, system_prompt, model, max_tokens, temperature, stream
                )

            # Handle non-streaming requests
            return self._handle_chat_request(
                messages, system_prompt, model, max_tokens, temperature
            )

        return await asyncio.to_thread(_execute)

    def _handle_streaming_request(
        self,
        messages: List[Mapping[str, str]],
        system_prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        stream: Callable[[str], None],
    ) -> LLMResult:
        """Handle streaming chat completion request."""
        tokens: List[str] = []

        for chunk in self._provider.stream_function(
            messages=list(messages),
            system=system_prompt,
            model_id=model,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            if not chunk:
                continue
            tokens.append(chunk)
            try:
                stream(chunk)
            except Exception:  # pragma: no cover - defensive logging only
                logger.exception("Stream handler raised while processing chunk")

        text = "".join(tokens)
        emit_event(
            self._provider.event_prefix,
            "llm_stream_completed",
            wrap_payload(model=model, tokens=len(tokens)),
        )
        return LLMResult(text=text)

    def _handle_chat_request(
        self,
        messages: List[Mapping[str, str]],
        system_prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResult:
        """Handle non-streaming chat completion request."""
        response_text = self._provider.chat_function(
            messages=list(messages),
            system=system_prompt,
            model_id=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        emit_event(
            self._provider.event_prefix,
            "llm_completed",
            wrap_payload(model=model, stream=False, tokens=len(response_text.split())),
        )
        return LLMResult(text=response_text)


class BedrockChatRunner(BaseChatRunner):
    """Runner issuing chat-completion requests to AWS Bedrock using aws_bedrock.py."""

    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        import os

        from dotenv import load_dotenv

        # Use bedrock_client.py with proper environment loading
        from bedrock_client import converse_claude, converse_claude_stream

        # Ensure environment is loaded
        load_dotenv()

        # Use bedrock_client.py functions directly
        def chat_function(messages, system, model_id, max_tokens, temperature):
            return converse_claude(
                messages=messages,
                system=system,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        def stream_function(messages, system, model_id, max_tokens, temperature):
            return converse_claude_stream(
                messages=messages,
                system=system,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        provider_config = LLMProviderConfig(
            event_prefix="bedrock_runner",
            stream_function=stream_function,
            chat_function=chat_function,
        )

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            provider_config=provider_config,
        )


class DeterministicLLMRunner(LLMRunner):
    """Runner that returns precomputed structured payloads for testing or demos."""

    def __init__(self, builder: Callable[[LLMRequest], Mapping[str, Any]]) -> None:
        self._builder = builder

    async def run(
        self,
        request: LLMRequest,
        *,
        stream: Optional[Callable[[str], None]] = None,
    ) -> LLMResult:
        data = self._builder(request)
        text = json.dumps(data)
        if stream is not None:
            stream(text)
        return LLMResult(text=text, usage={"mode": "deterministic"})


class BaseLLMAgent:
    """Base class providing LLM orchestration for incident agents with pluggable providers."""

    name: str

    def __init__(
        self,
        *,
        role: str,
        llm: Optional[LLMRunner] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        stream_updates: bool = True,
        max_validation_attempts: int = 2,
    ) -> None:
        self.name = role
        self._role = role
        self._llm_runner = llm
        self._model = model if model is not None else get_default_model()
        self._temperature = (
            temperature if temperature != 0.2 else get_default_temperature()
        )
        self._max_tokens = (
            max_output_tokens if max_output_tokens != 1024 else get_default_max_tokens()
        )
        self._stream_updates = stream_updates
        self._max_validation_attempts = max(1, int(max_validation_attempts))

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def set_llm_runner(self, runner: LLMRunner) -> None:
        """Override the runner used for subsequent LLM calls."""

        self._llm_runner = runner

    # ------------------------------------------------------------------
    # Core handle logic
    # ------------------------------------------------------------------
    async def handle(
        self,
        incoming: "AgentMessage",
        state: "IncidentState",
    ) -> "AgentMessage":
        system_prompt = self._system_prompt(incoming, state)
        user_prompt = self._build_user_prompt(incoming, state)
        messages: Sequence[Mapping[str, str]] = [
            {"role": "user", "content": user_prompt}
        ]

        runner = self._ensure_runner()
        last_error: Optional[Exception] = None
        raw_text: Optional[str] = None
        parsed: Optional[Mapping[str, Any]] = None

        for attempt in range(1, self._max_validation_attempts + 1):
            request = LLMRequest(
                system_prompt=system_prompt,
                messages=messages,
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                stream=self._stream_updates,
                metadata={"incoming": incoming, "state": state, "attempt": attempt},
            )
            try:
                result = await runner.run(request, stream=self._handle_stream_chunk)
                raw_text = result.text
                parsed = self._parse_response_text(result.text)
                break
            except Exception as exc:  # pragma: no cover - defensive path
                last_error = exc
                parsed = None

                # Enhanced retry logic with exponential backoff
                is_retryable = _is_retryable_error(exc)
                is_final_attempt = attempt >= self._max_validation_attempts

                logger.debug(
                    "LLM attempt %d/%d failed: %s (retryable=%s)",
                    attempt,
                    self._max_validation_attempts,
                    exc,
                    is_retryable,
                )

                # Don't retry if error is not retryable or this is the final attempt
                if not is_retryable or is_final_attempt:
                    if not is_retryable:
                        logger.info(
                            "Non-retryable error encountered, failing immediately: %s",
                            exc,
                        )
                    break

                # Calculate and apply exponential backoff delay
                if attempt < self._max_validation_attempts:
                    delay = _calculate_retry_delay(
                        attempt, base_delay=1.0, max_delay=60.0
                    )
                    logger.debug(
                        "Waiting %.2f seconds before retry %d", delay, attempt + 1
                    )
                    await asyncio.sleep(delay)

        if parsed is None:
            raise last_error or LLMResponseFormatError("LLM response parsing failed")

        message_type = self._message_type(parsed, incoming, state)
        payload = self._build_payload(parsed, incoming, state)

        message = self._build_message_instance(incoming, state, message_type, payload)
        # Attach rich LLM metadata to the message object for downstream consumers
        # (e.g., transcript logger and CSV export). This does not alter the
        # serialized payload; it's available at runtime and can be merged into
        # transcript metadata by the logger.
        try:
            meta = {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": raw_text or "",
                "model": self._model,
                "attempt": request.metadata.get("attempt", 1),
            }
            # Dynamic attribute for runtime access in UIs/exporters; bypass
            # pydantic setattr restrictions intentionally.
            object.__setattr__(message, "metadata", meta)
        except Exception:
            # Best-effort attachment; never break message flow on metadata issues
            logger.debug("Failed to attach LLM metadata to message", exc_info=True)
        self._after_message(
            message,
            parsed,
            incoming,
            state,
            used_fallback=False,
            raw_text=raw_text,
        )
        return message

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    def _system_prompt(self, incoming, state) -> str:
        raise NotImplementedError

    def _build_user_prompt(self, incoming, state) -> str:
        raise NotImplementedError

    def _message_type(self, parsed: Mapping[str, Any], incoming, state):
        raise NotImplementedError

    def _build_payload(self, parsed: Mapping[str, Any], incoming, state):
        raise NotImplementedError

    def _after_message(
        self,
        message,
        parsed: Mapping[str, Any],
        incoming,
        state,
        *,
        used_fallback: bool,
        raw_text: Optional[str],
    ) -> None:
        # Subclasses may override to emit observability events or update state.
        del message, parsed, incoming, state, used_fallback, raw_text

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_runner(self) -> LLMRunner:
        if self._llm_runner is None:
            provider = DEFAULT_LLM_PROVIDER.value.lower()
            if provider == "bedrock":
                self._llm_runner = BedrockChatRunner(
                    model=self._model,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            else:  # Default to Bedrock for simplified system
                self._llm_runner = BedrockChatRunner(
                    model=self._model,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
        return self._llm_runner

    def _build_message_instance(self, incoming, state, message_type, payload):
        from .schema import AgentMessage, AgentRole  # Local import to avoid cycles

        role = AgentRole(self._role)
        return AgentMessage(
            incident_id=incoming.incident_id,
            **{"from": role},
            **{"to": AgentRole.ORCHESTRATOR},
            type=message_type,
            severity=state.severity,
            payload=payload,
        )

    def _validate_and_repair_json(self, text: str) -> dict:
        """Validate JSON with repair attempts for common issues"""
        candidate = text.strip()
        if not candidate:
            raise LLMResponseFormatError("Empty LLM response")

        # Validate response size to prevent memory exhaustion
        if len(candidate) > MAX_JSON_RESPONSE_SIZE:
            raise LLMResponseFormatError(
                f"Response exceeds {MAX_JSON_RESPONSE_SIZE} bytes"
            )

        candidate = self._strip_code_fence(candidate)

        # Try direct parsing first
        try:
            data = json.loads(candidate)
            # Security: No expression evaluation - return JSON as-is
            return data
        except json.JSONDecodeError as exc:
            # The JSON might be followed by extra text - try to extract just the JSON part
            extracted_json = self._extract_valid_json(candidate)
            if extracted_json and extracted_json != candidate:
                try:
                    data = json.loads(extracted_json)
                    logger.info(
                        "Successfully extracted valid JSON from response with trailing text"
                    )
                    return data
                except json.JSONDecodeError:
                    pass

            # Log the ACTUAL response that failed to parse for debugging
            logger.error(
                "Failed to parse LLM JSON response. First 500 chars: %s...",
                candidate[:500],
            )
            logger.debug("Full failed response: %s", candidate)

            # Try common repairs
            repaired = self._attempt_json_repairs(candidate)
            if repaired:
                try:
                    data = json.loads(repaired)
                    # Security: No expression evaluation - return JSON as-is
                    return data
                except json.JSONDecodeError:
                    pass
            raise LLMResponseFormatError("LLM response was not valid JSON") from exc

    def _extract_valid_json(self, text: str) -> str | None:
        """Extract valid JSON from text that may have trailing content.

        This handles cases where the LLM returns valid JSON followed by extra text:
        {"key": "value"}
        To further explain...  <- Extra text that breaks parsing
        """
        # Track brace/bracket nesting level
        stack = []
        in_string = False
        escape_next = False

        for i, char in enumerate(text):
            # Handle string state (strings can contain { } [ ] that shouldn't be counted)
            if char == '"' and not escape_next:
                in_string = not in_string
            elif char == "\\" and in_string:
                escape_next = not escape_next
                continue
            else:
                escape_next = False

            # Skip characters inside strings
            if in_string:
                continue

            # Track nesting
            if char in "{[":
                stack.append(char)
            elif char in "}]":
                if not stack:
                    continue  # Unmatched closing bracket - ignore

                opening = stack.pop()
                # Check for mismatched brackets
                if (char == "}" and opening != "{") or (char == "]" and opening != "["):
                    # Mismatched - this isn't valid JSON
                    return None

                # If stack is empty, we've closed all braces - JSON is complete!
                if not stack:
                    # Extract everything up to and including this position
                    return text[: i + 1].strip()

        # If we get here, JSON wasn't properly closed
        return None

    def _attempt_json_repairs(self, text: str) -> str | None:
        """Attempt common JSON repairs"""
        import re

        repaired = text

        # Fix trailing commas before closing brackets/braces (most common LLM error)
        # Matches: ,] or ,} with any whitespace in between
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

        # Fix multiple consecutive commas
        repaired = re.sub(r",\s*,+", r",", repaired)

        # Fix unescaped quotes (basic attempt)
        # This is tricky - only do obvious cases
        repaired = repaired.replace('"', '"').replace('""', '"')

        # Fix incomplete arrays/objects at the end (if truncated)
        # Count opening and closing braces/brackets
        open_braces = repaired.count("{")
        close_braces = repaired.count("}")
        open_brackets = repaired.count("[")
        close_brackets = repaired.count("]")

        # Add missing closing braces
        if open_braces > close_braces:
            repaired += "}" * (open_braces - close_braces)

        # Add missing closing brackets
        if open_brackets > close_brackets:
            repaired += "]" * (open_brackets - close_brackets)

        return repaired

    def _parse_response_text(self, text: str) -> Mapping[str, Any]:
        """Parse LLM response with validation and repair attempts"""
        return self._validate_and_repair_json(text)

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Strip code fences and LLM artifacts from response text"""
        candidate = text.strip()

        # Remove common end-of-text tokens
        eot_tokens = ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "</s>"]
        for token in eot_tokens:
            if candidate.endswith(token):
                candidate = candidate[: -len(token)].strip()

        # Strip common LLM response prefixes that appear before JSON
        # Examples: "Summary: ...\n\nDetails (JSON):\n{...}", "Here's the JSON:\n{...}"
        lines = candidate.split("\n")
        json_start_idx = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            # Check if line starts with JSON (opening brace or bracket)
            if stripped.startswith("{") or stripped.startswith("["):
                json_start_idx = i
                break
            # Also check for "Details (JSON):" or similar markers
            if "Details" in stripped and ("JSON" in stripped or "json" in stripped):
                json_start_idx = i + 1
                break

        # If we found JSON start, extract from there
        if json_start_idx > 0:
            candidate = "\n".join(lines[json_start_idx:]).strip()

        # Handle code fences (```json, ```python, etc.)
        if candidate.startswith("```"):
            fence_lines = candidate.split("\n")
            if len(fence_lines) > 1:
                # Skip the first line (```json)
                start_idx = 1
                # Find the closing ``` from the end
                end_idx = len(fence_lines)
                for i in range(len(fence_lines) - 1, -1, -1):
                    if fence_lines[i].strip() == "```":
                        end_idx = i
                        break

                if end_idx > start_idx:
                    # Extract content between fences
                    body_lines = fence_lines[start_idx:end_idx]
                    return "\n".join(body_lines).strip()

        return candidate

    def _handle_stream_chunk(
        self, chunk: str
    ) -> None:  # pragma: no cover - hook for subclasses
        del chunk


__all__ = [
    "DeterministicLLMRunner",
    "BedrockChatRunner",
    "BaseLLMAgent",
    "LLMRequest",
    "LLMResult",
    "LLMResponseFormatError",
]
