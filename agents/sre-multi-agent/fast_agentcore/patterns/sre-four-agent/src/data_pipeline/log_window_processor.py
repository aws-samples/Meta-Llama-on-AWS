"""Log window processing utilities for extracting time-based windows from streaming logs."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


class LogEntry(BaseModel):
    """Structured representation of a streaming log entry."""

    timestamp: datetime
    service: str
    host: str
    pod: str
    trace_id: str
    request_id: str
    level: str
    message: str

    # Optional fields that may be present
    response_time_ms: Optional[int] = None
    processing_time_ms: Optional[int] = None
    error_code: Optional[str] = None
    http_status: Optional[int] = None
    transaction_id: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    query_time_ms: Optional[int] = None
    cpu_usage_pct: Optional[float] = None
    memory_usage_pct: Optional[float] = None
    concurrent_requests: Optional[int] = None

    # Additional metadata fields
    extra_fields: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_jsonl_line(cls, line: str) -> "LogEntry":
        """Create LogEntry from a JSONL line."""
        data = json.loads(line.strip())

        # Parse timestamp
        timestamp_str = data.pop("timestamp")
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

        # Extract known fields (handle both old and new data formats)
        known_fields = {
            "timestamp": timestamp,
            "service": data.pop("service"),
            "host": data.pop("host", data.pop("instance_id", "unknown-host")),
            "pod": data.pop("pod", data.pop("container_id", "unknown-pod")),
            "trace_id": data.pop("trace_id"),
            "request_id": data.pop("request_id"),
            "level": data.pop("level"),
            "message": data.pop("message"),
        }

        # Extract optional fields if present
        optional_fields = [
            "response_time_ms", "processing_time_ms", "error_code",
            "http_status", "transaction_id", "amount", "currency",
            "query_time_ms", "cpu_usage_pct", "memory_usage_pct",
            "concurrent_requests"
        ]

        for field in optional_fields:
            if field in data:
                known_fields[field] = data.pop(field)

        # Store remaining fields as extra
        known_fields["extra_fields"] = data

        return cls(**known_fields)


def load_streaming_logs(file_path: Path | str = "data/kafka_style/streaming_logs.jsonl") -> List[LogEntry]:
    """Load all streaming logs from the JSONL file."""
    logs = []
    file_path = Path(file_path)

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                logs.append(LogEntry.from_jsonl_line(line))

    # Sort by timestamp to ensure chronological order
    logs.sort(key=lambda x: x.timestamp)
    return logs


def extract_time_window(
    logs: List[LogEntry],
    start_time: datetime,
    window_minutes: int = 15
) -> List[LogEntry]:
    """
    Extract logs for a specific time window.

    Args:
        logs: List of log entries (should be sorted by timestamp)
        start_time: Start of the time window
        window_minutes: Duration of the window in minutes

    Returns:
        List of log entries within the specified time window
    """
    end_time = start_time + timedelta(minutes=window_minutes)

    window_logs = []
    for log in logs:
        if start_time <= log.timestamp <= end_time:
            window_logs.append(log)
        elif log.timestamp > end_time:
            # Since logs are sorted, we can break early
            break

    return window_logs


def get_window_metadata(logs: List[LogEntry]) -> Dict[str, Any]:
    """
    Extract metadata about a log window for analysis.

    Args:
        logs: List of log entries from the window

    Returns:
        Dictionary with window metadata
    """
    if not logs:
        return {
            "total_logs": 0,
            "services": [],
            "log_levels": {},
            "time_range": None,
            "error_count": 0,
            "warn_count": 0,
        }

    services = set()
    log_levels = {}
    error_count = 0
    warn_count = 0

    for log in logs:
        services.add(log.service)
        level = log.level
        log_levels[level] = log_levels.get(level, 0) + 1

        if level == "ERROR":
            error_count += 1
        elif level == "WARN":
            warn_count += 1

    return {
        "total_logs": len(logs),
        "services": sorted(list(services)),
        "log_levels": log_levels,
        "time_range": {
            "start": logs[0].timestamp.isoformat(),
            "end": logs[-1].timestamp.isoformat(),
        },
        "error_count": error_count,
        "warn_count": warn_count,
        "error_rate": round(error_count / len(logs) * 100, 2) if logs else 0.0,
        "warn_rate": round(warn_count / len(logs) * 100, 2) if logs else 0.0,
    }


def generate_window_schedule(
    start_time: datetime,
    total_hours: int = 5,
    window_minutes: int = 15
) -> List[datetime]:
    """
    Generate a schedule of time windows for the entire dataset.

    Args:
        start_time: Starting timestamp for the dataset
        total_hours: Total duration of the dataset in hours
        window_minutes: Duration of each window in minutes

    Returns:
        List of start times for each window
    """
    windows = []
    current_time = start_time
    end_time = start_time + timedelta(hours=total_hours)

    while current_time < end_time:
        windows.append(current_time)
        current_time += timedelta(minutes=window_minutes)

    return windows