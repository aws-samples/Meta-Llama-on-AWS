"""Orchestrator state helpers for the four-agent demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Set

from .schema import AgentMessage, AgentRole, Severity, MessageType


@dataclass
class IncidentState:
    """Mutable runtime state tracked by the orchestrator."""

    incident_id: str
    severity: Severity
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    timeline: List[AgentMessage] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    active_plan_id: Optional[str] = None
    plan_history: List[str] = field(default_factory=list)

    def add_message(self, message: AgentMessage) -> None:
        self.timeline.append(message)
        if (
            self.acknowledged_at is None
            and message.sender != AgentRole.ORCHESTRATOR
        ):
            self.acknowledged_at = message.ts
        if message.type == MessageType.STATUS and "state" in message.payload.details:
            if message.payload.details.get("state") == "resolved":
                self.resolved_at = message.ts

    def set_active_plan(self, plan_id: str) -> None:
        self.active_plan_id = plan_id
        if plan_id not in self.plan_history:
            self.plan_history.append(plan_id)


__all__ = ["IncidentState"]
