"""Configuration for Bedrock Knowledge Base integration.

This module provides centralized configuration management for Bedrock Knowledge Base
settings, loading values from environment variables with sensible defaults.

The BedrockKBConfig dataclass encapsulates all configuration needed for:
- Knowledge Base identification and data source management
- AWS region and service settings
- S3 bucket configuration for document storage
- Inference model selection (Llama 3.3/3.7)
- Retrieval parameters (top_k)

Configuration values are loaded with the following priority:
1. Environment variables (highest priority)
2. Default values (fallback)

Example usage:
    >>> config = BedrockKBConfig()
    >>> errors = config.validate()
    >>> if errors:
    ...     print(f"Configuration errors: {errors}")
    >>> else:
    ...     print(f"Using Knowledge Base: {config.knowledge_base_id}")
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available. Install with: pip install boto3")


@dataclass
class BedrockKBConfig:
    """Configuration for Bedrock Knowledge Base integration.
    
    This dataclass centralizes all configuration settings for the Bedrock Knowledge Base
    integration, loading values from environment variables with sensible defaults.
    
    Attributes:
        knowledge_base_id: Bedrock Knowledge Base ID (required)
        data_source_id: Data source ID for the Knowledge Base (required for sync operations)
        region: AWS region for Bedrock services (default: us-west-2)
        bucket_name: S3 bucket name for document storage (required)
        inference_model_arn: Model ARN for inference (default: Llama 3.3 70B)
        top_k: Number of results to retrieve from semantic search (default: 5)
    
    Environment Variables:
        BEDROCK_KB_ID: Knowledge Base ID
        BEDROCK_DATA_SOURCE_ID: Data source ID
        AWS_REGION: AWS region
        BEDROCK_KB_BUCKET: S3 bucket name
        BEDROCK_INFERENCE_MODEL: Model ARN for inference
        BEDROCK_KB_TOP_K: Number of results to retrieve
    
    Example:
        >>> # Load configuration from environment
        >>> config = BedrockKBConfig()
        >>> 
        >>> # Validate configuration
        >>> errors = config.validate()
        >>> if errors:
        ...     for error in errors:
        ...         print(f"Error: {error}")
        >>> 
        >>> # Use configuration
        >>> print(f"KB ID: {config.knowledge_base_id}")
        >>> print(f"Region: {config.region}")
        >>> print(f"Top K: {config.top_k}")
    """
    
    # Knowledge Base settings
    knowledge_base_id: str = ""
    data_source_id: str = ""
    
    # AWS settings
    region: str = "us-west-2"
    
    # S3 settings
    bucket_name: str = ""
    
    # Model settings
    inference_model_arn: str = (
        "arn:aws:bedrock:us-west-2::foundation-model/meta.llama3-3-70b-instruct-v1:0"
    )
    
    # Retrieval settings
    top_k: int = 5
    
    def __post_init__(self):
        """Load configuration from environment variables after initialization.
        
        This method is automatically called after the dataclass __init__ method.
        It loads configuration values from environment variables, using the
        dataclass field defaults as fallbacks.
        
        Environment variable loading priority:
        1. Environment variable value (if set)
        2. Automatic model selection via select_inference_model() (for inference_model_arn only)
        3. Dataclass field default (if env var not set)
        
        Note:
            This method modifies the instance attributes in-place based on
            environment variables. It does not validate the configuration;
            use the validate() method for validation.
        """
        # Load Knowledge Base settings from environment
        self.knowledge_base_id = os.getenv("BEDROCK_KB_ID", self.knowledge_base_id)
        self.data_source_id = os.getenv("BEDROCK_DATA_SOURCE_ID", self.data_source_id)
        
        # Load AWS settings from environment
        self.region = os.getenv("AWS_REGION", self.region)
        
        # Load S3 settings from environment
        self.bucket_name = os.getenv("BEDROCK_KB_BUCKET", self.bucket_name)
        
        # Load model settings from environment
        # If BEDROCK_INFERENCE_MODEL is set, use it (override)
        # Otherwise, use select_inference_model() to determine the best model
        env_model = os.getenv("BEDROCK_INFERENCE_MODEL")
        if env_model:
            self.inference_model_arn = env_model
            logger.info(
                f"Using inference model from BEDROCK_INFERENCE_MODEL: {env_model}"
            )
        else:
            # Use automatic model selection
            self.inference_model_arn = select_inference_model(self.region)
            logger.info(
                f"Using automatically selected inference model: {self.inference_model_arn}"
            )
        
        # Load retrieval settings from environment
        # Convert to int, using default if conversion fails
        try:
            top_k_str = os.getenv("BEDROCK_KB_TOP_K")
            if top_k_str:
                self.top_k = int(top_k_str)
        except (ValueError, TypeError):
            # Keep default value if conversion fails
            pass
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of error messages.
        
        Checks that all required configuration values are present and valid.
        Returns a list of error messages for any missing or invalid configuration.
        
        Required configuration:
        - knowledge_base_id: Must be non-empty string
        - bucket_name: Must be non-empty string
        
        Optional configuration (validated if present):
        - top_k: Must be >= 1
        - region: Must be non-empty string
        
        Returns:
            List of error messages. Empty list if configuration is valid.
            Each error message describes a specific configuration problem.
        
        Example:
            >>> config = BedrockKBConfig()
            >>> errors = config.validate()
            >>> if errors:
            ...     print("Configuration errors:")
            ...     for error in errors:
            ...         print(f"  - {error}")
            ... else:
            ...     print("Configuration is valid")
        """
        errors = []
        
        # Check required configuration
        if not self.knowledge_base_id:
            errors.append(
                "BEDROCK_KB_ID is required. "
                "Set environment variable BEDROCK_KB_ID to your Knowledge Base ID."
            )
        
        if not self.bucket_name:
            errors.append(
                "BEDROCK_KB_BUCKET is required. "
                "Set environment variable BEDROCK_KB_BUCKET to your S3 bucket name."
            )
        
        # Validate optional configuration if present
        if self.top_k < 1:
            errors.append(
                f"BEDROCK_KB_TOP_K must be >= 1, got {self.top_k}. "
                "Set environment variable BEDROCK_KB_TOP_K to a positive integer."
            )
        
        if not self.region:
            errors.append(
                "AWS_REGION is required. "
                "Set environment variable AWS_REGION to your AWS region (e.g., us-west-2)."
            )
        
        return errors


def check_llama_37_availability(region: str = "us-west-2") -> bool:
    """Check if Llama 3.7 (meta.llama3-3-70b-instruct-v1:0) is available in the region.
    
    Uses the Bedrock list_foundation_models() API to check if Llama 3.7 is available
    in the specified AWS region. This function is used to determine whether to use
    Llama 3.7 or fall back to Llama 3.3 70B for inference.
    
    Args:
        region: AWS region to check for model availability (default: us-west-2)
    
    Returns:
        True if Llama 3.7 is available in the region, False otherwise
    
    Raises:
        ImportError: If boto3 is not available
        ClientError: If there's an error calling the Bedrock API
    
    Example:
        >>> if check_llama_37_availability("us-west-2"):
        ...     print("Llama 3.7 is available")
        ... else:
        ...     print("Llama 3.7 is not available, using Llama 3.3 70B")
    
    Note:
        The function checks for the model ID "meta.llama3-3-70b-instruct-v1:0".
        This is the expected model ID for Llama 3.7 in AWS Bedrock.
        
        **Validates: Requirements 5.1**
    """
    if not BOTO3_AVAILABLE:
        raise ImportError(
            "boto3 is required to check model availability. "
            "Install with: pip install boto3"
        )
    
    try:
        # Initialize Bedrock client for the specified region
        bedrock_client = boto3.client("bedrock", region_name=region)
        
        # List all foundation models available in the region
        logger.debug(f"Checking Llama 3.7 availability in region: {region}")
        response = bedrock_client.list_foundation_models()
        
        # Check if Llama 3.7 model is in the list
        # Model ID for Llama 3.7: meta.llama3-3-70b-instruct-v1:0
        llama_37_model_id = "meta.llama3-3-70b-instruct-v1:0"
        
        for model in response.get("modelSummaries", []):
            model_id = model.get("modelId", "")
            if model_id == llama_37_model_id:
                logger.info(
                    f"Llama 3.7 (model ID: {llama_37_model_id}) is available in region {region}"
                )
                return True
        
        logger.info(
            f"Llama 3.7 (model ID: {llama_37_model_id}) is not available in region {region}"
        )
        return False
        
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_message = e.response.get("Error", {}).get("Message", str(e))
        
        logger.error(
            f"Failed to check Llama 3.7 availability in region {region}: "
            f"Code={error_code}, Message={error_message}"
        )
        raise
    
    except Exception as e:
        logger.error(
            f"Unexpected error checking Llama 3.7 availability in region {region}: "
            f"{type(e).__name__}: {e}"
        )
        raise


def select_inference_model(region: str = "us-west-2") -> str:
    """Select the best available inference model for the region.
    
    Implements the model selection logic with the following priority:
    1. Llama 3.7 (meta.llama3-3-70b-instruct-v1:0) if available
    2. Llama 3.3 70B (meta.llama3-3-70b-instruct-v1:0) as fallback
    3. Claude 3 Sonnet (anthropic.claude-3-sonnet-20240229-v1:0) as final fallback
    
    The function checks model availability using the Bedrock API and logs the
    selection decision. If the Bedrock API is unavailable or returns errors,
    it falls back to Claude 3 Sonnet.
    
    Args:
        region: AWS region to check for model availability (default: us-west-2)
    
    Returns:
        Model ARN string for the selected inference model
    
    Example:
        >>> model_arn = select_inference_model("us-west-2")
        >>> print(f"Selected model: {model_arn}")
        Selected model: arn:aws:bedrock:us-west-2::foundation-model/meta.llama3-3-70b-instruct-v1:0
    
    Note:
        This function handles errors gracefully and always returns a valid model ARN.
        If all checks fail, it defaults to Claude 3 Sonnet with appropriate logging.
        
        **Validates: Requirements 5.2, 5.3, 5.5**
    """
    # Model ARNs for different models
    llama_37_arn = f"arn:aws:bedrock:{region}::foundation-model/meta.llama3-3-70b-instruct-v1:0"
    llama_33_arn = f"arn:aws:bedrock:{region}::foundation-model/meta.llama3-3-70b-instruct-v1:0"
    claude_sonnet_arn = f"arn:aws:bedrock:{region}::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
    
    # Check if boto3 is available
    if not BOTO3_AVAILABLE:
        logger.warning(
            "boto3 not available. Falling back to Claude 3 Sonnet. "
            "Install boto3 to enable Llama model selection."
        )
        logger.info(f"Selected model: Claude 3 Sonnet (ARN: {claude_sonnet_arn})")
        return claude_sonnet_arn
    
    try:
        # Check if Llama 3.7 is available
        logger.debug(f"Checking for Llama 3.7 availability in region {region}")
        
        if check_llama_37_availability(region):
            logger.info(
                f"Selected model: Llama 3.7 (ARN: {llama_37_arn}) - "
                "Model is available in the region"
            )
            return llama_37_arn
        
        # Llama 3.7 not available, use Llama 3.3 70B
        logger.info(
            f"Selected model: Llama 3.3 70B (ARN: {llama_33_arn}) - "
            "Llama 3.7 not available in the region"
        )
        return llama_33_arn
        
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_message = e.response.get("Error", {}).get("Message", str(e))
        
        logger.error(
            f"Bedrock API error during model selection: "
            f"Code={error_code}, Message={error_message}. "
            f"Falling back to Claude 3 Sonnet."
        )
        logger.info(f"Selected model: Claude 3 Sonnet (ARN: {claude_sonnet_arn})")
        return claude_sonnet_arn
        
    except Exception as e:
        logger.error(
            f"Unexpected error during model selection: "
            f"{type(e).__name__}: {e}. "
            f"Falling back to Claude 3 Sonnet."
        )
        logger.info(f"Selected model: Claude 3 Sonnet (ARN: {claude_sonnet_arn})")
        return claude_sonnet_arn


__all__ = ["BedrockKBConfig", "check_llama_37_availability", "select_inference_model"]
