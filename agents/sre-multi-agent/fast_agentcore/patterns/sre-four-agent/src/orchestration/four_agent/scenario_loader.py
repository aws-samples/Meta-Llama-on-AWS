"""Helpers for loading deterministic scenario snapshots for the demo."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Iterable, Mapping
import uuid

from src.utils.time_window import IncidentWindow
from .schema import Severity


@dataclass(frozen=True)
class ScenarioMetadata:
    key: str
    severity: Severity
    description: str


SCENARIO_REGISTRY: Mapping[str, ScenarioMetadata] = {
    "sev3": ScenarioMetadata(
        key="sev3",
        severity=Severity.SEV_3,
        description="Memory leak / noisy neighbor on payments-api (node-b)",
    ),
    "sev2": ScenarioMetadata(
        key="sev2",
        severity=Severity.SEV_2,
        description="Regional issuer dependency failure impacting us-east",
    ),
    "sev1": ScenarioMetadata(
        key="sev1",
        severity=Severity.SEV_1,
        description="Payment pipeline disruption (Kafka / DB outage)",
    ),
}


@dataclass
class ScenarioSnapshot:
    """Aggregate of monitoring data and context for a predefined window."""

    metadata: ScenarioMetadata
    window: IncidentWindow
    monitoring: Dict[str, object]
    additional_sources: Dict[str, object]
    _incident_id: str | None = None  # Cached incident ID

    @property
    def incident_id(self) -> str:
        """Get or generate incident ID (cached after first access)."""
        if self._incident_id is None:
            today = date.today().strftime("%Y%m%d")
            scenario_id = str(uuid.uuid4())[:8].upper()
            # Cache the generated ID so it doesn't change on subsequent access
            object.__setattr__(self, '_incident_id', f"INC-{today}-{scenario_id}")
        return self._incident_id

    @classmethod
    def create_basic_scenario(
        cls,
        scenario_type: str,
        metadata: Dict[str, object] | None = None
    ) -> ScenarioSnapshot:
        """Create a basic scenario snapshot for API-triggered incidents.

        Args:
            scenario_type: Type of scenario (sev1, sev2, sev3, or custom)
            metadata: Additional metadata for the scenario

        Returns:
            ScenarioSnapshot instance for the requested scenario type
        """
        from datetime import datetime, timezone

        # Get or create scenario metadata
        scenario_key = scenario_type.lower()
        if scenario_key in SCENARIO_REGISTRY:
            scenario_metadata = SCENARIO_REGISTRY[scenario_key]
        else:
            # Create a basic custom scenario
            severity_map = {
                "sev1": Severity.SEV_1,
                "sev2": Severity.SEV_2,
                "sev3": Severity.SEV_3,
            }

            severity = severity_map.get(scenario_key, Severity.SEV_3)
            scenario_metadata = ScenarioMetadata(
                key=scenario_key,
                severity=severity,
                description=f"API-triggered {scenario_type} scenario"
            )

        # Create a basic incident window (current time window)
        now = datetime.now(timezone.utc)
        window = IncidentWindow(
            start=now.replace(minute=0, second=0, microsecond=0),
            end=now
        )

        # Create basic monitoring data structure
        monitoring = {
            "system_name": "api-triggered",
            "time_window": {
                "start": _iso(window.start),
                "end": _iso(window.end),
            },
            "metrics": {
                "source": "api",
                "triggered_at": _iso(now),
                "scenario_type": scenario_type,
                **(metadata or {})
            },
        }

        additional_sources = {
            "api_metadata": metadata or {},
            "creation_method": "api_basic_scenario"
        }

        return cls(
            metadata=scenario_metadata,
            window=window,
            monitoring=monitoring,
            additional_sources=additional_sources
        )


def _iso(ts: datetime) -> str:
    return ts.replace(microsecond=0).isoformat() + "Z"


def load_snapshot(key: str, data_dir: Path | str = Path("data")) -> ScenarioSnapshot:
    """DEPRECATED: Uses old CSV data loader that was removed.
    
    This function relied on src.data_ingestion which has been deleted.
    Use ScenarioSnapshot.create_basic_scenario() instead.
    """
    raise NotImplementedError(
        "load_snapshot() uses deprecated CSV data loader that was removed. "
        "Use ScenarioSnapshot.create_basic_scenario() instead."
    )


def load_snapshots(
    keys: Iterable[str] | None = None,
    *,
    data_dir: Path | str = Path("data"),
) -> Dict[str, ScenarioSnapshot]:
    """DEPRECATED: Uses old CSV data loader that was removed.
    
    This function relied on src.data_ingestion which has been deleted.
    Use ScenarioSnapshot.create_basic_scenario() instead.
    """
    raise NotImplementedError(
        "load_snapshots() uses deprecated CSV data loader that was removed. "
        "Use ScenarioSnapshot.create_basic_scenario() instead."
    )


# Alias for backward compatibility
TimeWindow = IncidentWindow

__all__ = [
    "ScenarioMetadata",
    "ScenarioSnapshot",
    "SCENARIO_REGISTRY",
    "TimeWindow",
    "load_snapshot",
    "load_snapshots",
]
