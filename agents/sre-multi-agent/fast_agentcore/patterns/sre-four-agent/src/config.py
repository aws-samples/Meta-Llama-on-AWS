"""Centralized configuration management for SRE POC.

This module provides a single source of truth for all configuration using Pydantic Settings.
Configuration can be loaded from:
1. Environment variables (.env file)
2. Configuration files (configs/*.json)
3. Runtime overrides

All magic numbers and thresholds are documented here with clear explanations.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AWSSettings(BaseSettings):
    """AWS-specific configuration."""

    model_config = SettingsConfigDict(
        env_prefix="AWS_", env_file=".env", extra="ignore"
    )

    region: str = Field(
        default="us-east-1", description="AWS region for Bedrock and other services"
    )
    access_key_id: str | None = Field(
        default=None, description="AWS Access Key ID (optional if using IAM roles)"
    )
    secret_access_key: str | None = Field(
        default=None, description="AWS Secret Access Key (optional if using IAM roles)"
    )


class BedrockSettings(BaseSettings):
    """AWS Bedrock LLM configuration."""

    model_config = SettingsConfigDict(
        env_prefix="BEDROCK_", env_file=".env", extra="ignore"
    )

    default_model: str = Field(
        default="us.meta.llama3-3-70b-instruct-v1:0",
        description="Default Bedrock model ID for LLM agents",
    )
    max_tokens: int = Field(
        default=2000, description="Maximum tokens for LLM responses"
    )
    temperature: float = Field(
        default=0.1,
        description="LLM temperature (0.0-1.0). Lower = more deterministic.",
    )
    embedding_model: str = Field(
        default="amazon.titan-embed-text-v1",
        description="Bedrock model ID for vector embeddings (RAG)",
    )


class AnalystAgentSettings(BaseSettings):
    """Configuration for Analyst Agent."""

    model_config = SettingsConfigDict(
        env_prefix="ANALYST_", env_file=".env", extra="ignore"
    )

    # Confidence thresholds
    high_confidence_threshold: float = Field(
        default=0.80,
        description="Threshold for high confidence anomaly detection (80%)",
    )
    medium_confidence_threshold: float = Field(
        default=0.60,
        description="Threshold for medium confidence anomaly detection (60%)",
    )
    orchestration_trigger_threshold: float = Field(
        default=0.75,
        description=(
            "Minimum confidence to trigger full incident response (75%). "
            "Below this threshold, anomalies are logged but don't trigger RCA/Impact/Mitigation."
        ),
    )

    # Log analysis parameters
    max_log_lines_analyzed: int = Field(
        default=500,
        description="Maximum number of log lines to analyze in a single window",
    )
    include_temporal_context: bool = Field(
        default=True, description="Include temporal patterns in analysis"
    )
    include_service_correlation: bool = Field(
        default=True, description="Analyze correlation between service failures"
    )


class PipelineSettings(BaseSettings):
    """Data pipeline configuration."""

    model_config = SettingsConfigDict(
        env_prefix="PIPELINE_", env_file=".env", extra="ignore"
    )

    # Time window configuration
    window_minutes: int = Field(
        default=2, description="Size of time window for log analysis (minutes)"
    )
    stream_interval_minutes: int = Field(
        default=1,
        description="Interval between streaming windows - creates overlap (minutes)",
    )

    # Log processing limits
    max_logs_per_window: int = Field(
        default=100,
        description=(
            "Maximum logs to send to LLM per window. "
            "Prevents token limit issues with large log volumes."
        ),
    )
    window_count: int = Field(
        default=10, description="Number of windows to process in a pipeline run"
    )

    # File paths
    log_file_path: str = Field(
        default="data/kafka_style/bank_logs.jsonl", description="Path to log data file"
    )
    incident_manifest_path: str = Field(
        default="data/kafka_style/manifest_incidents.yaml",
        description="Path to incident manifest file",
    )

    # Processing options
    enable_incident_detection: bool = Field(
        default=True, description="Enable automatic incident detection"
    )
    enable_log_aggregation: bool = Field(
        default=True, description="Aggregate similar log entries"
    )
    cache_logs: bool = Field(
        default=True, description="Cache loaded logs in memory for faster processing"
    )


class RAGSettings(BaseSettings):
    """RAG (Retrieval-Augmented Generation) configuration."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=".env", extra="ignore"
    )

    # Vector search configuration
    chunk_size: int = Field(
        default=500, description="Size of text chunks for embedding (characters)"
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between chunks to preserve context (characters)",
    )
    top_k: int = Field(
        default=5, description="Number of top results to retrieve from vector search"
    )

    # Cache configuration
    cache_dir: str = Field(
        default=".vector_cache", description="Directory to cache vector embeddings"
    )
    enable_cache: bool = Field(
        default=True, description="Enable vector store caching for faster startup"
    )

    # Policy document paths
    runbooks_path: str = Field(
        default="docs/policies/troubleshooting-runbooks.md",
        description="Path to troubleshooting runbooks (POL-SRE-003)",
    )
    patterns_path: str = Field(
        default="docs/policies/known-failure-patterns.md",
        description="Path to known failure patterns (POL-SRE-004)",
    )
    business_metrics_path: str = Field(
        default="docs/policies/business-impact-baselines.md",
        description="Path to business impact baselines (POL-SRE-002)",
    )
    incident_procedures_path: str = Field(
        default="docs/policies/incident-response-procedures.md",
        description="Path to incident response procedures (POL-SRE-001)",
    )


class OutputSettings(BaseSettings):
    """Output and reporting configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OUTPUT_", env_file=".env", extra="ignore"
    )

    # Output paths
    outputs_dir: str = Field(
        default="outputs", description="Directory for generated reports"
    )
    logs_dir: str = Field(default="logs", description="Directory for application logs")

    # Output options
    include_raw_logs: bool = Field(
        default=True, description="Include raw logs in reports"
    )
    include_metrics: bool = Field(
        default=True, description="Include performance metrics in reports"
    )
    include_incident_details: bool = Field(
        default=True, description="Include full incident details in reports"
    )

    # Data retention
    retention_days: int = Field(
        default=30, description="Number of days to retain output files before cleanup"
    )


class UISettings(BaseSettings):
    """Frontend UI configuration."""

    model_config = SettingsConfigDict(env_prefix="UI_", env_file=".env", extra="ignore")

    # Server configuration
    backend_host: str = Field(default="127.0.0.1", description="Backend server host")
    backend_port: int = Field(default=8000, description="Backend server port")
    frontend_port: int = Field(
        default=3000, description="Frontend development server port"
    )

    # WebSocket configuration
    enable_websocket: bool = Field(
        default=True, description="Enable WebSocket real-time updates"
    )
    websocket_path: str = Field(
        default="/ws/demo", description="WebSocket endpoint path"
    )

    # CORS configuration
    cors_origins: str = Field(
        default="*", description="Comma-separated list of allowed CORS origins"
    )


class Settings(BaseSettings):
    """Main application settings - aggregates all configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # Environment
    environment: str = Field(
        default="development",
        description="Environment: development, staging, or production",
    )
    debug: bool = Field(default=True, description="Enable debug mode")
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )

    # Sub-configurations
    aws: AWSSettings = Field(default_factory=AWSSettings)
    bedrock: BedrockSettings = Field(default_factory=BedrockSettings)
    analyst: AnalystAgentSettings = Field(default_factory=AnalystAgentSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    ui: UISettings = Field(default_factory=UISettings)

    def __init__(self, **kwargs):
        """Initialize settings and load from config files if they exist."""
        super().__init__(**kwargs)
        self._load_from_config_files()

    def _load_from_config_files(self):
        """Load additional configuration from JSON files."""
        # Load pipeline configuration from configs/data_pipeline.json
        pipeline_config_path = Path("configs/data_pipeline.json")
        if pipeline_config_path.exists():
            try:
                with open(pipeline_config_path) as f:
                    config_data = json.load(f)

                # Update pipeline settings from JSON file
                if "processing" in config_data:
                    processing = config_data["processing"]
                    if "window_minutes" in processing:
                        self.pipeline.window_minutes = processing["window_minutes"]
                    if "stream_interval_minutes" in processing:
                        self.pipeline.stream_interval_minutes = processing[
                            "stream_interval_minutes"
                        ]
                    if "max_logs_per_window" in processing:
                        self.pipeline.max_logs_per_window = processing[
                            "max_logs_per_window"
                        ]
                    if "window_count" in processing:
                        self.pipeline.window_count = processing["window_count"]

                # Update data sources
                if (
                    "data_sources" in config_data
                    and "local" in config_data["data_sources"]
                ):
                    local = config_data["data_sources"]["local"]
                    if "log_file" in local:
                        self.pipeline.log_file_path = local["log_file"]
                    if "incident_manifest_file" in local:
                        self.pipeline.incident_manifest_path = local[
                            "incident_manifest_file"
                        ]

            except Exception as e:
                # Don't fail on config file errors, just log warning
                import logging

                logging.warning(
                    f"Failed to load pipeline config from {pipeline_config_path}: {e}"
                )

    def get_full_log_path(self) -> Path:
        """Get absolute path to log file."""
        return Path(self.pipeline.log_file_path).resolve()

    def get_outputs_dir(self) -> Path:
        """Get absolute path to outputs directory."""
        path = Path(self.output.outputs_dir)
        path.mkdir(exist_ok=True)
        return path

    def get_logs_dir(self) -> Path:
        """Get absolute path to logs directory."""
        path = Path(self.output.logs_dir)
        path.mkdir(exist_ok=True)
        return path

    def get_vector_cache_dir(self) -> Path:
        """Get absolute path to vector cache directory."""
        path = Path(self.rag.cache_dir)
        path.mkdir(exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    This function is cached so the settings are only loaded once per process.
    Use this function throughout the application instead of creating Settings directly.

    Returns:
        Application settings instance
    """
    return Settings()


# Convenience functions for backward compatibility


def get_default_model() -> str:
    """Get default LLM model ID."""
    return get_settings().bedrock.default_model


def get_default_temperature() -> float:
    """Get default LLM temperature."""
    return get_settings().bedrock.temperature


def get_default_max_tokens() -> int:
    """Get default LLM max tokens."""
    return get_settings().bedrock.max_tokens


# Export all settings classes and functions
__all__ = [
    "Settings",
    "AWSSettings",
    "BedrockSettings",
    "AnalystAgentSettings",
    "PipelineSettings",
    "RAGSettings",
    "OutputSettings",
    "UISettings",
    "get_settings",
    "get_default_model",
    "get_default_temperature",
    "get_default_max_tokens",
]
