"""Simple time window utility."""
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class IncidentWindow:
    """Represents a time window for incident analysis."""
    start: datetime
    end: datetime
