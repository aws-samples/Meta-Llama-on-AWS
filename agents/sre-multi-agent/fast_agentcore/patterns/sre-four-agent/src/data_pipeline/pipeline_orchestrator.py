"""Main pipeline orchestrator for log data processing and scenario generation."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Any, AsyncIterator

from .log_window_processor import (
    load_streaming_logs,
    extract_time_window,
    generate_window_schedule,
    LogEntry
)


class LogDataPipeline:
    """
    Main log data pipeline for processing Kafka-style streaming logs.

    Provides both demo mode (using synthetic dataset) and prepares for
    real-time mode (OpenSearch integration).
    """

    def __init__(
        self,
        log_file_path: Path | str = None,
        window_minutes: int = None,
        stream_interval_minutes: Optional[int] = None,
        config_file: Path | str = "configs/data_pipeline.json"
    ):
        """
        Initialize the log data pipeline.

        Args:
            log_file_path: Path to the streaming logs JSONL file (overrides config)
            window_minutes: Duration of analysis windows in minutes (overrides config)
            stream_interval_minutes: How often to process new windows (overrides config)
            config_file: Path to pipeline configuration file
        """
        # Load configuration
        self.config = self._load_config(config_file)

        # Set parameters (command line/constructor args override config)
        self.log_file_path = Path(log_file_path or self.config["data_sources"]["local"]["log_file"])
        self.window_minutes = window_minutes or self.config["processing"]["window_minutes"]
        self.stream_interval_minutes = stream_interval_minutes or self.config["processing"]["stream_interval_minutes"]

        # Pipeline state
        self._logs_cache: Optional[List[LogEntry]] = None
        self._dataset_start_time: Optional[datetime] = None

        # Validate configuration
        self._validate_configuration()

    def _load_config(self, config_file: Path | str) -> Dict[str, Any]:
        """Load pipeline configuration from JSON file."""
        config_path = Path(config_file)

        # If config file doesn't exist, create default config
        if not config_path.exists():
            print(f"⚠️  Config file {config_path} not found, using default configuration")
            return self._get_default_config()

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            if config.get("logging", {}).get("show_pipeline_stats", True):
                print(f"📋 Loaded pipeline config from {config_path}")

            return config
        except Exception as e:
            print(f"❌ Error loading config from {config_path}: {e}")
            print("   Using default configuration")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default pipeline configuration."""
        return {
            "data_sources": {
                "local": {
                    "log_file": "data/kafka_style/bank_logs.jsonl",
                    "incident_metadata_file": "data/kafka_style/incident_metadata.json"
                }
            },
            "processing": {
                "window_minutes": 15,
                "stream_interval_minutes": 5,
                "max_logs_per_window": 1000
            },
            "logging": {
                "show_pipeline_stats": True,
                "show_window_overlap_info": True
            }
        }

    def _validate_configuration(self):
        """Validate pipeline configuration and show informational messages."""
        show_info = self.config.get("logging", {}).get("show_window_overlap_info", True)

        if not show_info:
            return

        if self.stream_interval_minutes > self.window_minutes:
            print(f"⚠️  Warning: Stream interval ({self.stream_interval_minutes}min) > window size ({self.window_minutes}min)")
            print("   This will create gaps between windows")
        elif self.stream_interval_minutes < self.window_minutes:
            overlap_minutes = self.window_minutes - self.stream_interval_minutes
            print(f"📊 Overlapping windows: {self.window_minutes}-min windows every {self.stream_interval_minutes} min ({overlap_minutes}-min overlap)")
        else:
            print(f"🎯 Standard windows: {self.window_minutes}-min windows every {self.stream_interval_minutes} min (no overlap)")

    def load_logs(self) -> List[LogEntry]:
        """Load streaming logs, using cache if available."""
        if self._logs_cache is None:
            self._logs_cache = load_streaming_logs(self.log_file_path)
            if self._logs_cache:
                self._dataset_start_time = self._logs_cache[0].timestamp

        return self._logs_cache

    @property
    def dataset_start_time(self) -> datetime:
        """Get the start time of the dataset."""
        if self._dataset_start_time is None:
            logs = self.load_logs()
            if logs:
                self._dataset_start_time = logs[0].timestamp
            else:
                # Fallback to known dataset start time
                from datetime import timezone
                self._dataset_start_time = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)

        return self._dataset_start_time

    def get_available_windows(self) -> List[datetime]:
        """Get all available time windows in the dataset."""
        logs = self.load_logs()
        if not logs:
            return []

        start_time = logs[0].timestamp
        end_time = logs[-1].timestamp
        total_hours = (end_time - start_time).total_seconds() / 3600

        return generate_window_schedule(start_time, int(total_hours) + 1, self.window_minutes)

    def get_window_logs(self, window_start: datetime) -> List[LogEntry]:
        """
        Get logs for a specific time window.

        Args:
            window_start: Start time of the window to process

        Returns:
            List of LogEntry objects for the specified window
        """
        logs = self.load_logs()
        return extract_time_window(logs, window_start, self.window_minutes)

    def get_window_summary(self, window_start: datetime) -> str:
        """
        Get a human-readable summary of a specific window.

        Args:
            window_start: Start time of the window

        Returns:
            Formatted summary string
        """
        window_logs = self.get_window_logs(window_start)

        lines = [
            f"=== WINDOW SUMMARY ===",
            f"Time: {window_start.strftime('%Y-%m-%d %H:%M:%S')} - "
            f"{(window_start + timedelta(minutes=self.window_minutes)).strftime('%H:%M:%S')} UTC",
            f"Logs: {len(window_logs)} entries",
        ]

        if window_logs:
            # Add service breakdown
            services = {}
            for log in window_logs:
                services[log.service] = services.get(log.service, 0) + 1

            lines.append(f"\nService Activity:")
            for service, count in sorted(services.items()):
                lines.append(f"  {service}: {count} logs")

        return "\n".join(lines)

    async def simulate_streaming_windows(
        self,
        start_time: Optional[datetime] = None,
        num_windows: int = 20,
        delay_seconds: float = 1.0
    ) -> AsyncIterator[List[LogEntry]]:
        """
        Simulate real-time streaming by yielding log windows sequentially.

        Args:
            start_time: Starting time for simulation (defaults to dataset start)
            num_windows: Number of windows to process
            delay_seconds: Delay between window processing (simulates real-time)

        Yields:
            List of LogEntry objects for each time window
        """
        if start_time is None:
            start_time = self.dataset_start_time

        current_time = start_time

        for i in range(num_windows):
            window_logs = self.get_window_logs(current_time)
            yield window_logs

            # Advance by stream interval (not window size) for overlapping windows
            current_time += timedelta(minutes=self.stream_interval_minutes)

            # Add delay to simulate real-time processing
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

