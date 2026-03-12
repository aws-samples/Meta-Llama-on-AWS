"""RCA agent powered by Groq LLM completions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from ..observability import emit_event, wrap_payload

# Evidence generator removed for simplified demo
from .llm import BaseLLMAgent
from .rca_knowledge_reader import (
    RCAKnowledgeReader,
)  # Fallback for when vector deps unavailable
from .schema import AgentRole, EvidenceReference, MessageType, PayloadModel

# Import Bedrock Knowledge Base reader for semantic search
try:
    from .bedrock_kb_reader import BedrockKnowledgeBaseReader

    BEDROCK_KB_AVAILABLE = True
except ImportError:
    BEDROCK_KB_AVAILABLE = False


@dataclass
class Hypothesis:
    """Structured representation of an RCA hypothesis used for fallbacks."""

    text: str
    confidence: float
    evidence: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "hypothesis": self.text,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
        }


class RCAAgent(BaseLLMAgent):
    """Produces root-cause hypotheses with Groq-backed reasoning."""

    name = AgentRole.RCA.value

    def __init__(
        self,
        *,
        llm=None,
        model: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1100,
        stream_updates: bool = True,
    ) -> None:
        # Use Bedrock Knowledge Base for semantic search
        if BEDROCK_KB_AVAILABLE:
            try:
                self._rca_knowledge_reader = BedrockKnowledgeBaseReader()
                print("✅ RCA Agent: Using Bedrock Knowledge Base (semantic search)")
            except Exception as e:
                print(
                    f"⚠️  RCA Agent: Bedrock KB initialization failed ({e}), falling back to keyword-based"
                )
                self._rca_knowledge_reader = RCAKnowledgeReader()
        else:
            print(
                "ℹ️  RCA Agent: Bedrock KB dependencies not available, using keyword-based RAG"
            )
            self._rca_knowledge_reader = RCAKnowledgeReader()

        super().__init__(
            role=AgentRole.RCA.value,
            llm=llm,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            stream_updates=stream_updates,
        )

    # ------------------------------------------------------------------
    # BaseLLMAgent hooks
    # ------------------------------------------------------------------
    def _extract_error_patterns(self, signals: dict) -> list[str]:
        """Extract error patterns from signals context.

        Args:
            signals: Signals context dictionary containing anomalies and error patterns

        Returns:
            List of error pattern strings
        """
        anomalies = signals.get("anomalies", [])
        error_patterns = signals.get("error_patterns", [])

        # Combine both sources
        all_patterns = []

        # Extract pattern strings from anomaly objects
        if isinstance(anomalies, list):
            for anomaly in anomalies:
                if isinstance(anomaly, dict):
                    # Anomaly is a dict with 'pattern' field
                    pattern = anomaly.get("pattern", "")
                    if pattern:
                        all_patterns.append(pattern)
                elif isinstance(anomaly, str):
                    # Anomaly is already a string
                    all_patterns.append(anomaly)

        # Add error patterns (should already be strings)
        if isinstance(error_patterns, list):
            for pattern in error_patterns:
                if isinstance(pattern, str):
                    all_patterns.append(pattern)

        return all_patterns

    def _system_prompt(self, incoming, state) -> str:
        return (
            "You are the RCA agent in an incident response team. "
            "Analyse telemetry to produce ranked root-cause hypotheses with evidence references."
        )

    def _build_user_prompt(self, incoming, state) -> str:
        monitoring = incoming.payload.details.get("monitoring", {})
        additional = incoming.payload.details.get("additional_sources", {})
        signals_context = incoming.payload.details.get("signals", {})

        # RAG: Extract error patterns from signals
        error_patterns = self._extract_error_patterns(signals_context)

        # RAG: Retrieve troubleshooting knowledge for the first error pattern
        troubleshooting_knowledge = {}
        if error_patterns:
            troubleshooting_knowledge = (
                self._rca_knowledge_reader.get_troubleshooting_steps(error_patterns[0])
            )

        # RAG: Retrieve known failure patterns
        # Try to infer service from error patterns or use a generic lookup
        service_hint = "database"  # Default fallback
        for pattern in error_patterns:
            if "payment" in pattern.lower() or "gateway" in pattern.lower():
                service_hint = "payment"
                break
            elif "memory" in pattern.lower() or "oom" in pattern.lower():
                service_hint = "memory"
                break

        failure_patterns = self._rca_knowledge_reader.get_failure_pattern(service_hint)

        # RAG: Retrieve similar past incidents based on error patterns (not raw anomalies)
        similar_incidents = []
        if error_patterns:
            similar_incidents = self._rca_knowledge_reader.get_similar_incidents(
                error_patterns  # Use extracted string patterns, not dict anomalies
            )

        # RAG: Get error code guidance for the first error pattern
        error_code_guidance = {}
        if error_patterns:
            error_code_guidance = self._rca_knowledge_reader.get_error_code_guidance(
                error_patterns[0]
            )

        context = {
            "incident_id": incoming.incident_id,
            "severity": getattr(state.severity, "value", str(state.severity)),
            "metrics": monitoring.get("metrics", {}),
            "infra": additional.get("infra", {}),
            "app": additional.get("app", {}),
            "signals_context": signals_context,
            "business": additional.get("business", {}),
            # RAG-enhanced knowledge
            "troubleshooting_knowledge": troubleshooting_knowledge,
            "failure_patterns": failure_patterns,
            "similar_incidents": similar_incidents,
            "error_code_guidance": error_code_guidance,
            "policy_reference": "POL-SRE-003 Troubleshooting Runbooks, POL-SRE-004 Known Failure Patterns",
        }
        schema = {
            "summary": "Short summary highlighting the top hypothesis and severity.",
            "details": {
                "ranked_hypotheses": [
                    {
                        "hypothesis": "string",
                        "confidence": "0-1 float",
                        "evidence": "object of metric:value pairs",
                        "recommended_actions": "optional list of steps",
                    }
                ],
                "signals_context": "echo of relevant signals",
                "business_context": "echo of business telemetry",
            },
            "evidence": "List of evidence objects with title/href/summary",
        }
        return (
            "Analyse the context and generate EXACTLY 3 root-cause hypotheses ranked by confidence. "
            "Each hypothesis should be a distinct potential cause, not variations of the same issue. "
            "Respond with JSON and populate the requested fields.\n"
            "Context:```json\n"
            f"{json.dumps(context, indent=2)}\n`````\n"
            "Required JSON schema:```json\n"
            f"{json.dumps(schema, indent=2)}\n`````"
        )

    def _message_type(
        self, parsed: Mapping[str, object], incoming, state
    ) -> MessageType:
        return MessageType.HYPOTHESIS

    def _build_payload(
        self, parsed: Mapping[str, object], incoming, state
    ) -> PayloadModel:
        details = parsed.get("details") or {}
        summary = parsed.get("summary")

        # Generate simple evidence references from monitoring data (simplified demo)
        evidence = None
        try:
            # Extract metrics from the incoming message for evidence
            monitoring = incoming.payload.details.get("monitoring", {})
            metrics = monitoring.get("metrics", {}) or {}

            # Create simple evidence references without external file generation
            evidence = [
                EvidenceReference(
                    title="Monitoring Metrics",
                    summary=(
                        f"Key metrics observed: {', '.join(list(metrics.keys())[:3])}"
                        if metrics
                        else "No metrics available"
                    ),
                ),
                EvidenceReference(
                    title="RCA Analysis Data",
                    summary="Root cause analysis based on incident patterns and system behavior",
                ),
            ]
        except Exception:
            # Gracefully fall back to empty evidence if generation fails
            evidence = None

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
        ranked = message.payload.details.get("ranked_hypotheses", [])
        top = ranked[0]["hypothesis"] if ranked else None
        emit_event(
            "agent.rca",
            "agent_completed",
            wrap_payload(
                incident_id=message.incident_id,
                severity=message.severity.value if message.severity else None,
                top_hypothesis=top,
                used_fallback=used_fallback,
            ),
        )

    # ------------------------------------------------------------------
    # Fallback logic
    # ------------------------------------------------------------------
