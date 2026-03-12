"""Configuration settings for four-agent orchestration."""

import logging
import os
from enum import Enum
from typing import Union

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""

    BEDROCK = "bedrock"


class ConfigurationError(ValueError):
    """Raised when configuration values are invalid."""

    pass


def _validate_int(
    env_var: str, default: int, min_val: int = 1, max_val: int = 100000
) -> int:
    """Validate integer environment variable with bounds checking.

    Args:
        env_var: Environment variable name
        default: Default value if env var not set
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Validated integer value

    Raises:
        ConfigurationError: If value is invalid or out of bounds
    """
    raw_value = os.getenv(env_var)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as e:
        raise ConfigurationError(
            f"Invalid integer value for {env_var}='{raw_value}'. Must be an integer."
        ) from e

    if not (min_val <= value <= max_val):
        raise ConfigurationError(
            f"Value for {env_var}={value} is out of bounds. "
            f"Must be between {min_val} and {max_val}."
        )

    return value


def _validate_float(
    env_var: str, default: float, min_val: float = 0.0, max_val: float = 2.0
) -> float:
    """Validate float environment variable with bounds checking.

    Args:
        env_var: Environment variable name
        default: Default value if env var not set
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Validated float value

    Raises:
        ConfigurationError: If value is invalid or out of bounds
    """
    raw_value = os.getenv(env_var)
    if raw_value is None:
        return default

    try:
        value = float(raw_value)
    except ValueError as e:
        raise ConfigurationError(
            f"Invalid float value for {env_var}='{raw_value}'. Must be a number."
        ) from e

    if not (min_val <= value <= max_val):
        raise ConfigurationError(
            f"Value for {env_var}={value} is out of bounds. "
            f"Must be between {min_val} and {max_val}."
        )

    return value


def _validate_provider(env_var: str, default: str) -> LLMProvider:
    """Validate LLM provider name.

    Args:
        env_var: Environment variable name
        default: Default provider name

    Returns:
        Validated LLMProvider enum

    Raises:
        ConfigurationError: If provider name is invalid
    """
    raw_value = os.getenv(env_var, default).lower().strip()

    try:
        return LLMProvider(raw_value)
    except ValueError as e:
        valid_providers = [p.value for p in LLMProvider]
        raise ConfigurationError(
            f"Invalid LLM provider '{raw_value}' for {env_var}. "
            f"Valid options: {valid_providers}"
        ) from e


# LLM Provider Configuration - Bedrock only
DEFAULT_LLM_PROVIDER = LLMProvider.BEDROCK

# AWS Bedrock Configuration (validated)
BEDROCK_DEFAULT_MODEL = os.getenv(
    "BEDROCK_DEFAULT_MODEL", "us.meta.llama3-3-70b-instruct-v1:0"
)
BEDROCK_DEFAULT_MAX_TOKENS = _validate_int(
    "BEDROCK_DEFAULT_MAX_TOKENS", 4096, min_val=1, max_val=32768
)
BEDROCK_DEFAULT_TEMPERATURE = _validate_float(
    "BEDROCK_DEFAULT_TEMPERATURE", 0.1, min_val=0.0, max_val=2.0
)

# Response size limits (validated)
MAX_JSON_RESPONSE_SIZE = _validate_int(
    "MAX_JSON_RESPONSE_SIZE", 100000, min_val=1024, max_val=10_000_000
)  # 1KB-10MB

# Business calculation defaults (validated)
BASELINE_TPS_MULTIPLIER = _validate_float(
    "BASELINE_TPS_MULTIPLIER", 1.0, min_val=0.1, max_val=10.0
)
BASELINE_REVENUE_MULTIPLIER = _validate_float(
    "BASELINE_REVENUE_MULTIPLIER", 1.0, min_val=0.1, max_val=10.0
)
REVENUE_PER_TRANSACTION = _validate_float(
    "REVENUE_PER_TRANSACTION", 2.45, min_val=0.01, max_val=1000.0
)
REVENUE_PER_APPROVAL = _validate_float(
    "REVENUE_PER_APPROVAL", 3.20, min_val=0.01, max_val=1000.0
)


def get_default_model() -> str:
    """Get the default model based on the selected provider with validation."""
    if DEFAULT_LLM_PROVIDER == LLMProvider.BEDROCK:
        return BEDROCK_DEFAULT_MODEL
    else:  # Fallback to Bedrock (Groq no longer default)
        return BEDROCK_DEFAULT_MODEL


def get_default_max_tokens() -> int:
    """Get the default max tokens based on the selected provider with validation."""
    if DEFAULT_LLM_PROVIDER == LLMProvider.BEDROCK:
        return BEDROCK_DEFAULT_MAX_TOKENS
    else:  # Fallback to Bedrock (Groq no longer default)
        return BEDROCK_DEFAULT_MAX_TOKENS


def get_default_temperature() -> float:
    """Get the default temperature based on the selected provider with validation."""
    if DEFAULT_LLM_PROVIDER == LLMProvider.BEDROCK:
        return BEDROCK_DEFAULT_TEMPERATURE
    else:  # Fallback to Bedrock (Groq no longer default)
        return BEDROCK_DEFAULT_TEMPERATURE


def validate_all_settings() -> None:
    """Validate all configuration settings at startup.

    Raises:
        ConfigurationError: If any configuration value is invalid
    """
    try:
        logger.info("Validating configuration settings...")

        # Validate provider configuration
        logger.debug(f"LLM Provider: {DEFAULT_LLM_PROVIDER.value}")
        logger.debug(f"Default Model: {get_default_model()}")
        logger.debug(f"Max Tokens: {get_default_max_tokens()}")
        logger.debug(f"Temperature: {get_default_temperature()}")

        # Validate size limits
        logger.debug(f"Max JSON Response Size: {MAX_JSON_RESPONSE_SIZE}")

        # Validate business calculations
        logger.debug(f"TPS Multiplier: {BASELINE_TPS_MULTIPLIER}")
        logger.debug(f"Revenue Multiplier: {BASELINE_REVENUE_MULTIPLIER}")
        logger.debug(f"Revenue per Transaction: {REVENUE_PER_TRANSACTION}")
        logger.debug(f"Revenue per Approval: {REVENUE_PER_APPROVAL}")

        logger.info("Configuration validation completed successfully")

    except ConfigurationError as e:
        logger.error("Configuration validation failed: %s", e)
        raise
