"""Enhanced observability helpers for agent workflow telemetry with CloudWatch integration."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Mapping, Optional

# CloudWatch integration removed for simplified demo
CLOUDWATCH_AVAILABLE = False

_LOGGER_NAME = "sre.observability"
_DEFAULT_FILENAME = "observability.log"


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    log_dir = os.getenv("SRE_POC_LOG_DIR", "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, _DEFAULT_FILENAME)
        handler = logging.FileHandler(log_path)
    except OSError:
        handler = logging.StreamHandler()

    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


_OBS_LOGGER = _configure_logger()


def emit_event(source: str, event: str, payload: Optional[Mapping[str, Any]] = None) -> None:
    """Emit a structured observability event as a JSON log line and CloudWatch metrics."""
    # Skip observability during visual demos to keep output clean
    if os.getenv("SRE_DEMO_VISUAL_MODE") == "1":
        return

    record = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "source": source,
        "event": event,
        "payload": payload or {},
    }

    # Emit to file/console logger
    try:
        _OBS_LOGGER.info(json.dumps(record, default=str))
    except TypeError:
        _OBS_LOGGER.info(json.dumps(record, default=_stringify))

    # Also emit to CloudWatch if available
    if CLOUDWATCH_AVAILABLE and os.getenv("ENABLE_CLOUDWATCH_OBSERVABILITY", "true").lower() == "true":
        _emit_to_cloudwatch(source, event, payload or {})

def wrap_payload(**kwargs: Any) -> Dict[str, Any]:
    """Shallow helper to build payload dictionaries with automatic str conversion."""
    return {key: _stringify(value) for key, value in kwargs.items() if value is not None}


def _stringify(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_stringify(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): _stringify(value) for key, value in obj.items()}
    return str(obj)


def _emit_to_cloudwatch(source: str, event: str, payload: Dict[str, Any]) -> None:
    """Emit observability events as CloudWatch metrics and structured logs."""
    try:
        # Map source to component name
        component_mapping = {
            "agent.analyst": "AnalystAgent",
            "opensearch": "OpenSearchIntegration",
            "system": "SystemHealth",
        }

        component = component_mapping.get(source, source.replace(".", "_").title())

        # Get observability instance
        obs = get_observability(component)

        # Extract incident/trace IDs from payload
        incident_id = payload.get("incident_id")
        trace_id = payload.get("trace_id")

        # Set context if available
        if incident_id or trace_id:
            obs.set_trace_context(trace_id=trace_id, incident_id=incident_id)

        # Log the event with structured data
        obs.log_info(
            message=f"Event: {event}",
            details={"source": source, "event": event, **payload}
        )

        # Emit specific metrics based on event type and source
        _emit_specific_metrics(obs, source, event, payload)

    except Exception as e:
        # Don't let CloudWatch errors break the application
        logging.getLogger(__name__).debug(f"Failed to emit to CloudWatch: {e}")


def _emit_specific_metrics(obs: Any, source: str, event: str, payload: Dict[str, Any]) -> None:
    """Emit specific metrics based on event type."""
    try:
        # Analyst Agent metrics
        if source == "agent.analyst":
            if event == "agent_completed":
                confidence = payload.get("anomaly_confidence", 0.0)
                severity_score = payload.get("severity_score", 0.0)

                obs.emit_anomaly_metrics(
                    anomalies_detected=1 if confidence > 0.5 else 0,
                    confidence_avg=confidence,
                    detection_time_ms=1000.0  # Default, would be better to track actual time
                )

            elif event == "autonomous_incident_triggered":
                confidence = payload.get("confidence", 0.0)
                obs.emit_metric("AutonomousIncidentsTriggered", 1.0, "Count")
                obs.emit_metric("TriggerConfidence", confidence, "None")

        # Multi-incident coordinator metrics
        elif source == "multi_incident_coordinator":
            if event == "incident_started":
                active_count = payload.get("active_count", 0)
                obs.emit_incident_metrics(
                    incidents_created=1,
                    concurrent_active=active_count,
                    completion_time_s=0.0,
                    success_rate=100.0
                )

            elif event == "incident_completed":
                message_count = payload.get("message_count", 0)
                obs.emit_metric("IncidentMessageCount", float(message_count), "Count")

            elif event == "incident_failed":
                obs.emit_metric("IncidentFailures", 1.0, "Count")

        # OpenSearch metrics
        elif "opensearch" in source.lower():
            if "query" in event.lower():
                latency = payload.get("latency_ms", payload.get("response_time_ms", 0))
                success = payload.get("success", True)
                result_count = payload.get("result_count", 0)

                obs.emit_opensearch_metrics(
                    query_latency_ms=float(latency),
                    success_rate=100.0 if success else 0.0,
                    logs_processed=result_count
                )

        # System health metrics
        elif source in ["system", "coordinator"]:
            if event in ["coordinator_started", "coordinator_shutdown"]:
                obs.emit_metric("SystemEvents", 1.0, "Count", {"EventType": event})

    except Exception as e:
        # Don't let metric emission errors break the application
        logging.getLogger(__name__).debug(f"Failed to emit specific metrics: {e}")


# Enhanced observability functions for direct CloudWatch integration
def emit_anomaly_detection_metrics(
    anomalies_detected: int,
    confidence_avg: float,
    detection_time_ms: float,
    component: str = "AnalystAgent"
) -> None:
    """Emit anomaly detection metrics directly to CloudWatch."""
    if CLOUDWATCH_AVAILABLE:
        try:
            obs = get_observability(component)
            obs.emit_anomaly_metrics(anomalies_detected, confidence_avg, detection_time_ms)
        except Exception as e:
            logging.getLogger(__name__).debug(f"Failed to emit anomaly metrics: {e}")


def emit_incident_processing_metrics(
    incidents_created: int = 0,
    concurrent_active: int = 0,
    completion_time_s: float = 0.0,
    success_rate: float = 100.0,
    component: str = "MultiIncidentCoordinator"
) -> None:
    """Emit incident processing metrics directly to CloudWatch."""
    if CLOUDWATCH_AVAILABLE:
        try:
            obs = get_observability(component)
            obs.emit_incident_metrics(incidents_created, concurrent_active, completion_time_s, success_rate)
        except Exception as e:
            logging.getLogger(__name__).debug(f"Failed to emit incident metrics: {e}")


def emit_opensearch_metrics(
    query_latency_ms: float,
    success_rate: float,
    logs_processed: int,
    component: str = "OpenSearchIntegration"
) -> None:
    """Emit OpenSearch metrics directly to CloudWatch."""
    if CLOUDWATCH_AVAILABLE:
        try:
            obs = get_observability(component)
            obs.emit_opensearch_metrics(query_latency_ms, success_rate, logs_processed)
        except Exception as e:
            logging.getLogger(__name__).debug(f"Failed to emit OpenSearch metrics: {e}")
