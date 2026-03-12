"""Incident summary utilities for four-agent orchestration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .schema import AgentMessage, AgentRole, MessageType, Severity
from .scenario_loader import ScenarioSnapshot
from .state import IncidentState


@dataclass
class IncidentSummary:
    """Aggregate statistics describing an incident run."""

    incident_id: str
    severity: Severity
    scenario: str
    opened_at: datetime
    acknowledged_at: datetime | None
    resolved_at: datetime | None
    mtta_minutes: float | None
    mttr_minutes: float | None
    total_revenue_loss: float | None
    approvals: List[Dict[str, object]]
    plan_id: str | None
    plan_revisions: int
    message_counts: Dict[str, int]
    status_summaries: List[str]

    def to_markdown(self) -> str:
        """Render the summary as a markdown document."""

        def _fmt(ts: datetime | None) -> str:
            if ts is None:
                return "—"
            return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

        def _fmt_minutes(value: float | None) -> str:
            if value is None:
                return "—"
            return f"{value:.1f} min"

        lines = [
            f"# Incident Summary – {self.incident_id}",
            "",
            f"* **Severity:** {self.severity.value}",
            f"* **Scenario:** {self.scenario}",
            f"* **Plan ID:** {self.plan_id or '—'} (revisions: {self.plan_revisions})",
            "",
            "## Timeline",
            f"- Opened: {_fmt(self.opened_at)}",
            f"- Acknowledged: {_fmt(self.acknowledged_at)} (MTTA {_fmt_minutes(self.mtta_minutes)})",
            f"- Resolved: {_fmt(self.resolved_at)} (MTTR {_fmt_minutes(self.mttr_minutes)})",
            "",
            "## Impact",
            f"- Estimated revenue loss: ${self.total_revenue_loss:.2f}"
            if self.total_revenue_loss is not None
            else "- Estimated revenue loss: —",
            "",
            "## Approvals",
        ]

        if self.approvals:
            for approval in self.approvals:
                status = "Approved" if approval.get("approved") else "Rejected"
                note = approval.get("note")
                note_section = f" — {note}" if note else ""
                lines.append(
                    f"- {approval.get('role')}: {status}{note_section}"
                )
        else:
            lines.append("- No approvals recorded")

        lines.extend(
            [
                "",
                "## Message Counts",
            ]
        )

        for msg_type, count in sorted(self.message_counts.items()):
            lines.append(f"- {msg_type}: {count}")

        if self.status_summaries:
            lines.extend(["", "## Status Updates"])
            for summary in self.status_summaries:
                lines.append(f"- {summary}")

        lines.append("")
        return "\n".join(lines)


class SummaryExporter:
    """Write incident summaries to markdown artifacts."""

    def __init__(self, base_dir: Path | str = Path("logs/summaries")) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def write(self, summary: IncidentSummary) -> Path:
        path = self._base_dir / f"{summary.incident_id}.md"
        path.write_text(summary.to_markdown(), encoding="utf-8")
        return path


def _default_acknowledged(state: IncidentState) -> datetime | None:
    if state.acknowledged_at:
        return state.acknowledged_at
    for message in state.timeline:
        if message.sender != AgentRole.ORCHESTRATOR:
            return message.ts
    return None


def _default_resolved(state: IncidentState) -> datetime | None:
    if state.resolved_at:
        return state.resolved_at
    for message in reversed(state.timeline):
        if message.type == MessageType.STATUS:
            return message.ts
    return None


def _minutes_between(start: datetime, end: datetime | None) -> float | None:
    if end is None:
        return None
    delta = end - start
    return round(delta.total_seconds() / 60.0, 2)


def _impact_loss(responses: Sequence[AgentMessage], duration_minutes: float) -> float | None:
    for message in responses:
        if message.type == MessageType.IMPACT:
            loss_per_min = message.payload.details.get(
                "estimated_revenue_loss_per_min"
            )
            if loss_per_min is None:
                return None
            return round(float(loss_per_min) * duration_minutes, 2)
    return None


def _approval_summary(approvals: Iterable[AgentMessage], plan_id: str | None) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    for message in approvals:
        if plan_id and message.payload.details.get("plan_id") != plan_id:
            continue
        summary.append(
            {
                "role": message.sender.value,
                "approved": bool(message.payload.details.get("approve")),
                "note": message.payload.details.get("note"),
            }
        )
    return summary


def _status_summaries(status_updates: Sequence[AgentMessage]) -> List[str]:
    summaries: List[str] = []
    for message in status_updates:
        summary = message.payload.summary
        if summary:
            summaries.append(summary)
        else:
            state = message.payload.details.get("state")
            summaries.append(f"Status update ({state})")
    return summaries


def build_incident_summary(
    state: IncidentState,
    responses: Sequence[AgentMessage],
    plans: Sequence[AgentMessage],
    approvals: Sequence[AgentMessage],
    status_updates: Sequence[AgentMessage],
    snapshot: ScenarioSnapshot,
) -> IncidentSummary:
    """Construct a summary object from orchestration artefacts."""

    acknowledged = _default_acknowledged(state)
    resolved = _default_resolved(state)
    mtta = _minutes_between(state.opened_at, acknowledged)
    mttr = _minutes_between(state.opened_at, resolved)

    duration_minutes = (snapshot.window.end - snapshot.window.start).total_seconds() / 60.0
    total_loss = _impact_loss(responses, duration_minutes)

    plan_id = state.active_plan_id
    if not plan_id and plans:
        plan_id = plans[-1].payload.details.get("plan_id")

    approvals_summary = _approval_summary(approvals, plan_id)
    status_summary = _status_summaries(status_updates)

    counts = Counter(message.type.value for message in state.timeline)

    return IncidentSummary(
        incident_id=state.incident_id,
        severity=state.severity,
        scenario=snapshot.metadata.description,
        opened_at=state.opened_at,
        acknowledged_at=acknowledged,
        resolved_at=resolved,
        mtta_minutes=mtta,
        mttr_minutes=mttr,
        total_revenue_loss=total_loss,
        approvals=approvals_summary,
        plan_id=plan_id,
        plan_revisions=len(state.plan_history),
        message_counts=dict(sorted(counts.items())),
        status_summaries=status_summary,
    )


__all__ = ["IncidentSummary", "SummaryExporter", "build_incident_summary"]

