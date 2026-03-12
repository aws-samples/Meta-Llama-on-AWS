"""Impact agent leveraging Groq LLM responses."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from ..observability import emit_event, wrap_payload
from .business_metrics_reader import BusinessMetricsReader
from .llm import BaseLLMAgent
from .schema import (AgentRole, EvidenceReference, MessageType, PayloadModel,
                     Severity)

# Import Bedrock Knowledge Base reader for semantic search
try:
    from .bedrock_kb_reader import BedrockKnowledgeBaseReader

    BEDROCK_KB_AVAILABLE = True
except ImportError:
    BEDROCK_KB_AVAILABLE = False


def _parse_window_minutes(window: dict[str, str] | None) -> float:
    if not window:
        return 0.0
    start_raw = window.get("start")
    end_raw = window.get("end")
    if not start_raw or not end_raw:
        return 0.0
    start = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
    end = datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
    return max((end - start).total_seconds() / 60.0, 0.0)


@dataclass(frozen=True)
class ImpactAgentConfig:
    """Configures baseline expectations for the impact agent."""

    baseline_success_rates: dict[Severity, float] = field(
        default_factory=lambda: {
            Severity.SEV_3: 98.5,
            Severity.SEV_2: 99.0,
            Severity.SEV_1: 99.5,
        }
    )


class ImpactAgent(BaseLLMAgent):
    """Computes TPS, approvals, and revenue deltas via Groq completions."""

    name = AgentRole.IMPACT.value

    def __init__(
        self,
        config: ImpactAgentConfig | None = None,
        *,
        llm=None,
        model: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 900,
        stream_updates: bool = True,
    ) -> None:
        self._config = config or ImpactAgentConfig()
        
        # Use Bedrock Knowledge Base for semantic search
        if BEDROCK_KB_AVAILABLE:
            try:
                self._business_metrics_reader = BedrockKnowledgeBaseReader()
                print("✅ Impact Agent: Using Bedrock Knowledge Base (semantic search)")
            except Exception as e:
                print(
                    f"⚠️  Impact Agent: Bedrock KB initialization failed ({e}), falling back to keyword-based"
                )
                self._business_metrics_reader = BusinessMetricsReader()
        else:
            print(
                "ℹ️  Impact Agent: Bedrock KB dependencies not available, using keyword-based RAG"
            )
            self._business_metrics_reader = BusinessMetricsReader()
        
        super().__init__(
            role=AgentRole.IMPACT.value,
            llm=llm,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            stream_updates=stream_updates,
        )

    # ------------------------------------------------------------------
    # BaseLLMAgent hooks
    # ------------------------------------------------------------------
    def _system_prompt(self, incoming, state) -> str:
        return (
            "You are the impact analyst on the incident team. "
            "Quantify customer approvals, TPS, and revenue deltas from telemetry."
        )

    def _build_user_prompt(self, incoming, state) -> str:
        monitoring = incoming.payload.details.get("monitoring", {})
        additional = incoming.payload.details.get("additional_sources", {})
        rca_context = incoming.payload.details.get("rca", {})

        # RAG: Retrieve baseline metrics from policy document using semantic search
        # For Bedrock KB, we use semantic search instead of specialized methods
        severity_str = getattr(state.severity, "value", str(state.severity))
        
        # Construct semantic queries for business metrics
        baseline_query = f"business impact baselines for {severity_str} severity incidents"
        revenue_query = "revenue calculation formulas and methodology"
        sla_query = f"SLA thresholds and breach criteria for {severity_str}"
        
        # Retrieve using semantic search
        try:
            # Check if we're using Bedrock KB by checking for the specific method
            if BEDROCK_KB_AVAILABLE and hasattr(self._business_metrics_reader, 'search_by_semantic_query') and not hasattr(self._business_metrics_reader, 'get_baseline_metrics'):
                # Use Bedrock KB semantic search
                baseline_results = self._business_metrics_reader.search_by_semantic_query(baseline_query)
                revenue_results = self._business_metrics_reader.search_by_semantic_query(revenue_query)
                sla_results = self._business_metrics_reader.search_by_semantic_query(sla_query)
                
                # Extract content from search results
                baseline_metrics = {
                    "retrieved_content": "\n".join([chunk["content"] for chunk in baseline_results[:2]]),
                    "policy_reference": "POL-SRE-002 Business Impact Baselines (via Bedrock KB)"
                }
                revenue_formulas = {
                    "retrieved_content": "\n".join([chunk["content"] for chunk in revenue_results[:2]]),
                    "policy_reference": "POL-SRE-002 Revenue Methodology (via Bedrock KB)"
                }
                sla_thresholds = {
                    "retrieved_content": "\n".join([chunk["content"] for chunk in sla_results[:2]]),
                    "policy_reference": "POL-SRE-002 SLA Thresholds (via Bedrock KB)"
                }
            else:
                # Fallback to BusinessMetricsReader methods
                baseline_metrics = self._business_metrics_reader.get_baseline_metrics(state.severity)
                revenue_formulas = self._business_metrics_reader.get_revenue_formulas()
                sla_thresholds = self._business_metrics_reader.get_sla_thresholds(state.severity)
        except Exception as e:
            print(f"⚠️  Impact Agent: RAG retrieval failed ({e}), using fallback")
            # Fallback to BusinessMetricsReader if available
            if hasattr(self._business_metrics_reader, 'get_baseline_metrics'):
                baseline_metrics = self._business_metrics_reader.get_baseline_metrics(state.severity)
                revenue_formulas = self._business_metrics_reader.get_revenue_formulas()
                sla_thresholds = self._business_metrics_reader.get_sla_thresholds(state.severity)
            else:
                # Ultimate fallback with empty data
                baseline_metrics = {"policy_reference": "RAG unavailable"}
                revenue_formulas = {"policy_reference": "RAG unavailable"}
                sla_thresholds = {"policy_reference": "RAG unavailable"}

        context = {
            "incident_id": incoming.incident_id,
            "severity": severity_str,
            "metrics": monitoring.get("metrics", {}),
            "time_window": monitoring.get("time_window"),
            "business": additional.get("business", {}),
            "signals": incoming.payload.details.get("signals", {}),
            "rca": rca_context,
            # RAG: Inject baseline metrics from policy document (POL-SRE-002)
            "business_baselines": baseline_metrics,
            "revenue_formulas": revenue_formulas,
            "sla_thresholds": sla_thresholds,
        }
        schema = {
            "summary": "Narrative describing approvals/TPS/revenue deltas.",
            "details": {
                "tps": {"actual": "float", "baseline": "float", "delta": "float"},
                "approvals": {"actual": "float", "baseline": "float", "delta": "float"},
                "revenue_per_min": {"actual": "float", "baseline": "float"},
                "estimated_revenue_loss_per_min": "float",
                "status_breakdown": "object",
            },
            "evidence": "List of evidence references",
        }
        return (
            "Assess business impact for the incident using the provided metrics. "
            "IMPORTANT: Use the documented baseline metrics from POL-SRE-002 (provided in business_baselines). "
            "Do NOT estimate or hallucinate baseline values - they are authoritative policy data. "
            "Apply the revenue calculation formulas from the policy to compute accurate revenue loss.\n"
            "Respond with JSON using the schema.\n"
            "Context:```json\n"
            f"{json.dumps(context, indent=2)}\n`````\n"
            "Required JSON schema:```json\n"
            f"{json.dumps(schema, indent=2)}\n`````"
        )

    def _message_type(
        self, parsed: Mapping[str, object], incoming, state
    ) -> MessageType:
        return MessageType.IMPACT

    def _build_payload(
        self, parsed: Mapping[str, object], incoming, state
    ) -> PayloadModel:
        summary = parsed.get("summary")
        details = parsed.get("details") or {}
        evidence_raw = parsed.get("evidence") or []
        evidence = [
            EvidenceReference(**item)
            for item in evidence_raw
            if isinstance(item, Mapping)
        ]
        return PayloadModel(
            summary=str(summary) if summary is not None else None,
            details=dict(details),
            evidence=evidence or None,
        )

    def _after_message(
        self,
        message,
        parsed: Mapping[str, object],
        incoming,
        state,
        *,
        used_fallback: bool,
        raw_text: str | None,
    ) -> None:
        details = message.payload.details
        # Simple validation for demo (business calculator removed)
        if not details:
            emit_event(
                "agent.impact",
                "validation_warning",
                wrap_payload(
                    incident_id=message.incident_id,
                    reason="Business impact metrics failed consistency validation",
                ),
            )
        emit_event(
            "agent.impact",
            "agent_completed",
            wrap_payload(
                incident_id=message.incident_id,
                severity=message.severity.value if message.severity else None,
                approvals_delta=details.get("approvals", {}).get("delta"),
                revenue_delta=details.get("estimated_revenue_loss_per_min"),
                used_fallback=used_fallback,
            ),
        )

    # ------------------------------------------------------------------
    # Fallback logic
    # ------------------------------------------------------------------
