"""Message schema and enumerations for the four-agent orchestration demo."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentRole(str, Enum):
    """Enumerates all participants that can post to the incident thread."""

    ORCHESTRATOR = "Orchestrator"
    SIGNALS = "Analyst"  # Renamed from "Signals" to "Analyst"
    ANALYST = "Analyst"  # Alias to SIGNALS (same value for routing compatibility)
    RCA = "RCA"
    IMPACT = "Impact"
    MITIGATION = "Mitigation"
    COMMS = "Comms"


class MessageType(str, Enum):
    """Supported types of messages exchanged in the thread."""

    OPEN = "open"
    REQUEST = "request"
    HYPOTHESIS = "hypothesis"
    IMPACT = "impact"
    PLAN = "plan"
    STATUS = "status"
    NOTE = "note"


class Severity(str, Enum):
    """Incident severity ladder used across scenarios."""

    SEV_3 = "SEV-3"
    SEV_2 = "SEV-2"
    SEV_1 = "SEV-1"


class EvidenceReference(BaseModel):
    """Lightweight pointer to supporting artefacts rendered in the UI."""

    title: str = Field(..., description="Human readable label for the evidence snippet.")
    href: Optional[str] = Field(
        None, description="Optional link or identifier for the evidence artefact."
    )
    summary: Optional[str] = Field(
        None, description="Short bullet summarising why the evidence matters."
    )


class PayloadModel(BaseModel):
    """Generic payload container with optional evidence references."""

    summary: Optional[str] = Field(
        None, description="Optional natural language summary for quick scanning."
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured key/value data shared with downstream agents.",
    )
    evidence: Optional[List[EvidenceReference]] = Field(
        default=None, description="Optional evidence objects supporting the message."
    )


class AgentMessage(BaseModel):
    """Canonical representation of a message exchanged between agents."""

    incident_id: str = Field(..., min_length=1)
    sender: AgentRole = Field(..., alias="from")
    recipient: Optional[AgentRole] = Field(None, alias="to", description="Target agent for this message")
    type: MessageType
    severity: Optional[Severity] = Field(
        None, description="Severity level associated with the message context."
    )
    payload: PayloadModel = Field(default_factory=PayloadModel)
    ts: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the message was emitted.",
    )

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def _validate_severity(self) -> "AgentMessage":
        required = {
            MessageType.OPEN,
            MessageType.HYPOTHESIS,
            MessageType.IMPACT,
            MessageType.PLAN,
            MessageType.STATUS,
            MessageType.REQUEST,
        }
        if self.type in required and self.severity is None:
            raise ValueError(
                f"severity is required for message type {self.type.value}"
            )
        return self


class ApprovalDecision(BaseModel):
    """Structured payload carried within approval messages."""

    plan_id: str = Field(..., min_length=1)
    approve: bool = Field(..., description="True if the plan is approved; False otherwise.")
    approvers: List[str] = Field(
        default_factory=list,
        description="Identities of approvers (IC, Risk, etc.).",
    )
    note: Optional[str] = Field(None, description="Optional justification or comments.")


class TranscriptEntry(BaseModel):
    """Structure persisted to the transcript for each message."""

    message: AgentMessage
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional runtime metadata (latency, etc.)."
    )


class DemoIncidentReport(BaseModel):
    """Structured output for AWS Bedrock demo presentations."""

    # Executive Summary
    executive_summary: str = Field(..., description="High-level incident summary for executives")
    severity_level: str = Field(..., description="SEV-1, SEV-2, or SEV-3")
    estimated_impact: str = Field(..., description="Business and user impact assessment")

    # Technical Details
    anomaly_details: Dict[str, Any] = Field(..., description="Raw anomaly detection results")
    root_cause_analysis: Dict[str, Any] = Field(..., description="RCA findings with hypotheses")
    technical_mitigation_plan: str = Field(..., description="Detailed technical remediation steps")

    # Communications
    stakeholder_notification: str = Field(..., description="Executive/internal stakeholder message")
    customer_communication: str = Field(..., description="Customer-facing status update")

    # Meta Information
    agents_involved: List[str] = Field(default_factory=list, description="List of agents that participated")
    rag_documents_used: List[str] = Field(default_factory=list, description="Policy documents referenced")
    processing_time_seconds: float = Field(default=0.0, description="Total processing time")
    incident_id: str = Field(..., description="Unique incident identifier")
    timestamp: str = Field(..., description="Incident start timestamp")


__all__ = [
    "AgentRole",
    "MessageType",
    "Severity",
    "EvidenceReference",
    "PayloadModel",
    "AgentMessage",
    "ApprovalDecision",
    "TranscriptEntry",
    "DemoIncidentReport",
]
