"""Mitigation and communications agent backed by Groq completions."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Mapping

from ..observability import emit_event, wrap_payload
# Business calculator removed for simplified demo
from .llm import BaseLLMAgent
from .policy_reader import PolicyReader
from .schema import AgentRole, EvidenceReference, MessageType, PayloadModel, Severity
from .settings import get_default_model

# Import Bedrock Knowledge Base reader for semantic search
try:
    from .bedrock_kb_reader import BedrockKnowledgeBaseReader

    BEDROCK_KB_AVAILABLE = True
except ImportError:
    BEDROCK_KB_AVAILABLE = False


@dataclass(frozen=True)
class MitigationAgentConfig:
    """Configuration for mitigation plan generation."""
    pass


class MitigationCommsAgent(BaseLLMAgent):
    """Drafts mitigation plans, approvals notes, and status updates via Groq."""

    name = AgentRole.MITIGATION.value

    def __init__(
        self,
        config: MitigationAgentConfig | None = None,
        *,
        llm=None,
        model: str | None = None,
        temperature: float = 0.25,
        max_output_tokens: int = 1400,
        stream_updates: bool = True,
    ) -> None:
        self._config = config or MitigationAgentConfig()
        self._plan_revisions: Dict[str, int] = defaultdict(int)
        self._cached_plan_details: Dict[str, Dict[str, object]] = {}
        # Business calculator removed for simplified demo
        
        # Use Bedrock Knowledge Base for semantic search, fallback to PolicyReader
        if BEDROCK_KB_AVAILABLE:
            try:
                self._kb_reader = BedrockKnowledgeBaseReader()
                print("✅ Mitigation Agent: Using Bedrock Knowledge Base (semantic search)")
            except Exception as e:
                print(
                    f"⚠️  Mitigation Agent: Bedrock KB initialization failed ({e}), falling back to PolicyReader"
                )
                self._kb_reader = None
        else:
            print(
                "ℹ️  Mitigation Agent: Bedrock KB dependencies not available, using PolicyReader"
            )
            self._kb_reader = None
        
        # Always keep PolicyReader as fallback
        self._policy_reader = PolicyReader()
        
        super().__init__(
            role=AgentRole.MITIGATION.value,
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
            "You are an expert SRE mitigation and communications agent for incident response operations. "
            "\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. Generate SPECIFIC, ACTIONABLE steps with exact commands, service names, and technical details\n"
            "2. Use REAL information from the RCA findings and impact assessment - no generic placeholders\n"
            "3. Reference specific error codes, service names, and infrastructure components from the analysis\n"
            "4. Include exact kubectl/docker/infrastructure commands with proper parameters\n"
            "5. Specify timings, thresholds, and success criteria for each step\n"
            "6. Draft targeted communications that mention specific findings and metrics\n"
            "\n"
            "BAD EXAMPLE: 'Restart payment service pods'\n"
            "GOOD EXAMPLE: 'kubectl rollout restart deployment/payment-processor-api -n production (Expected: 3/3 pods ready in 45s, verify via metrics dashboard)'\n"
            "\n"
            "BAD EXAMPLE: 'Monitor system health'\n"
            "GOOD EXAMPLE: 'Monitor TPS recovery: baseline 100 tps → current 45 tps → target 95+ tps within 5 minutes (Grafana dashboard: prod-payments-health)'\n"
            "\n"
            "You coordinate but do not execute changes directly. Your plans are handed off to execution teams. "
            "Always provide actionable, specific guidance with real data from the incident analysis."
        )

    def _build_user_prompt(self, incoming, state) -> str:
        details = incoming.payload.details
        intent = details.get("intent", "plan")
        incident_id = incoming.incident_id
        plan_context = self._cached_plan_details.get(incident_id)
        if intent not in {"plan", "revision", "status"}:
            intent = "plan"

        # Retrieve relevant policy procedures and mitigation playbooks
        severity_str = getattr(state.severity, "value", str(state.severity))
        
        # Try to use Bedrock KB for semantic search, fallback to PolicyReader
        if self._kb_reader is not None:
            try:
                # Use semantic search for mitigation playbooks and communication templates
                playbook_query = f"mitigation playbooks and procedures for {severity_str} severity incidents"
                communication_query = f"incident communication templates and stakeholder notification procedures for {severity_str}"
                approval_query = f"approval requirements and authorization procedures for {severity_str} incidents"
                
                # Retrieve using Bedrock KB semantic search
                playbook_results = self._kb_reader.search_by_semantic_query(playbook_query)
                communication_results = self._kb_reader.search_by_semantic_query(communication_query)
                approval_results = self._kb_reader.search_by_semantic_query(approval_query)
                
                # Extract content from search results
                policy_procedures = "\n".join([chunk["content"] for chunk in playbook_results[:2]])
                communication_templates = "\n".join([chunk["content"] for chunk in communication_results[:2]])
                approval_requirements = "\n".join([chunk["content"] for chunk in approval_results[:1]])
                
                # Add policy reference
                policy_reference = "POL-SRE-004 Mitigation Playbooks (via Bedrock KB)"
                
            except Exception as e:
                print(f"⚠️  Mitigation Agent: Bedrock KB retrieval failed ({e}), using PolicyReader fallback")
                # Fallback to PolicyReader
                policy_procedures = self._policy_reader.get_severity_procedures(state.severity)
                approval_requirements = self._policy_reader.get_approval_requirements(state.severity)
                communication_templates = ""
                policy_reference = "POL-SRE-001 Incident Response Procedures (PolicyReader fallback)"
        else:
            # Use PolicyReader when Bedrock KB is not available
            policy_procedures = self._policy_reader.get_severity_procedures(state.severity)
            approval_requirements = self._policy_reader.get_approval_requirements(state.severity)
            communication_templates = ""
            policy_reference = "POL-SRE-001 Incident Response Procedures (PolicyReader)"
        
        base_context = {
            "incident_id": incident_id,
            "severity": severity_str,
            "intent": intent,
            "signals": details.get("signals"),
            "rca": details.get("rca"),
            "impact": details.get("impact"),
            "feedback": details.get("feedback"),
            "prior_messages": details.get("prior_messages", []),
            "policy_procedures": policy_procedures,
            "approval_requirements": approval_requirements,
            "communication_templates": communication_templates,
            "policy_reference": policy_reference,
        }

        if intent == "status":
            context = {
                **base_context,
                "status_state": details.get("status_state", "executing"),
                "plan": details.get("plan") or plan_context,
                "approvals": details.get("approvals", []),
            }
        else:
            current_revision = self._plan_revisions.get(incident_id, 0)
            expected_revision = (
                current_revision + 1 if intent == "revision" else current_revision
            )
            expected_plan_id = self._format_plan_id(incident_id, expected_revision)
            context = {
                **base_context,
                "expected_plan_id": expected_plan_id,
                "expected_revision": expected_revision,
                "last_plan": details.get("last_plan") or plan_context,
            }

        schema = {
            "message_type": "'plan' or 'status'",
            "summary": "Short description of the plan or status update.",
            "details": "Structured object containing plan/status fields (see below).",
            "evidence": "Optional list of evidence references.",
        }
        plan_details = {
            "plan_id": "string matching expected_plan_id",
            "revision": "integer revision number",
            "status": "draft or ready_for_execution",
            "approvals_required": "list of approver roles",
            "objectives": "list of objectives",
            "steps": "ordered list of step objects",
            "communications": "internal/external communication drafts",
            "top_hypothesis": "string",
            "impact_snapshot": "object",
            "monitoring_summary": "object",
        }
        status_details = {
            "plan_id": "string",
            "coordination_state": "Current coordination state (coordination_active or handoff_complete)",
            "coordination_progress": "COORDINATION ONLY - describe team notifications and handoffs, NOT execution results",
            "team_handoffs": "list of which teams have been notified and their acknowledgment status",
            "next_coordination_action": "what coordination step happens next",
            "next_update_eta": "string",
            "communication": "map of coordination messages sent to teams",
        }
        return (
            "Generate comprehensive mitigation plans and coordination updates for incident response. "
            "\n\n"
            "CRITICAL INSTRUCTIONS FOR MITIGATION STEPS:\n"
            "- Extract service names, error codes, and infrastructure details from the RCA and impact analysis\n"
            "- Use SPECIFIC metrics from the impact assessment (e.g., TPS values, revenue numbers, error rates)\n"
            "- Reference ACTUAL log patterns and error messages from the analyst findings\n"
            "- Include exact commands with service names from the context (kubectl, docker, systemctl, etc.)\n"
            "- Specify monitoring queries and dashboard names for verification\n"
            "- Set concrete success criteria based on the impact metrics (e.g., 'TPS returns to baseline 100+/sec')\n"
            "\n"
            "EXAMPLE OF GOOD MITIGATION STEP:\n"
            "{\n"
            '  "action": "kubectl rollout restart deployment/payment-processor-api -n production",\n'
            '  "objective": "Clear connection pool exhaustion (current: 45 TPS vs baseline 100 TPS, $1250/min revenue loss)",\n'
            '  "owner": "SRE-Team-Payments",\n'
            '  "success_criteria": "All 3 pods healthy, TPS >95, error rate <1%, verify in Grafana: prod-payments-health",\n'
            '  "rollback_procedure": "kubectl rollout undo deployment/payment-processor-api if TPS drops or errors spike"\n'
            "}\n"
            "\n"
            "Use the provided policy procedures and mitigation playbooks as guidance for structured incident response. "
            "If communication templates are provided, use them as a starting point for drafting stakeholder communications. "
            "If intent is plan/revision, provide detailed mitigation plans with clear objectives and steps. "
            "If intent is status, report coordination progress and team handoff status. "
            "Reference relevant procedures and best practices in your response. "
            "Respond with JSON only.\n"
            "Context:```json\n"
            f"{json.dumps(context, indent=2)}\n`````\n"
            "Plan details schema (when message_type='plan'):```json\n"
            f"{json.dumps(plan_details, indent=2)}\n`````\n"
            "Status details schema (when message_type='status'):```json\n"
            f"{json.dumps(status_details, indent=2)}\n`````\n"
            "Top-level schema:```json\n"
            f"{json.dumps(schema, indent=2)}\n`````"
        )

    def _message_type(
        self, parsed: Mapping[str, object], incoming, state
    ) -> MessageType:
        intent = str(parsed.get("message_type", "plan")).lower()
        if intent == "status":
            return MessageType.STATUS
        return MessageType.PLAN

    def _build_payload(
        self, parsed: Mapping[str, object], incoming, state
    ) -> PayloadModel:
        summary = parsed.get("summary")
        details = parsed.get("details") or {}
        evidence_raw = parsed.get("evidence") or []
        evidence = [
            self._transform_evidence_item(item)
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
        incident_id = message.incident_id
        if message.type == MessageType.PLAN:
            details = dict(message.payload.details)
            plan_id = details.get("plan_id")
            revision = int(details.get("revision") or 0)
            self._plan_revisions[incident_id] = revision
            self._cached_plan_details[incident_id] = details
            emit_event(
                "agent.mitigation",
                "plan_generated",
                wrap_payload(
                    incident_id=incident_id,
                    plan_id=plan_id,
                    revision=revision,
                    approvals=list(details.get("approvals_required", [])),
                    used_fallback=used_fallback,
                ),
            )
        elif message.type == MessageType.STATUS:
            details = message.payload.details
            emit_event(
                "agent.mitigation",
                "status_published",
                wrap_payload(
                    incident_id=incident_id,
                    plan_id=details.get("plan_id"),
                    state=details.get("state"),
                    used_fallback=used_fallback,
                ),
            )

    def _format_plan_id(self, incident_id: str, revision: int) -> str:
        base = incident_id.replace("INC-", "PLAN-")
        if revision:
            return f"{base}-r{revision}"
        return f"{base}-primary"

    def _status_details(
        self, plan: Mapping[str, object], status_state: str
    ) -> Dict[str, object]:
        steps = plan.get("steps", []) if isinstance(plan, Mapping) else []
        communications = (
            plan.get("communications", {}) if isinstance(plan, Mapping) else {}
        )
        if status_state == "resolved":
            summary = f"Plan {plan.get('plan_id')} completed; success metrics back within SLO."
            return {
                "summary": summary,
                "completed_steps": [
                    step.get("action") for step in steps if isinstance(step, Mapping)
                ],
                "pending_steps": [],
                "communication": {
                    "external": communications.get("external"),
                    "internal": communications.get("internal"),
                },
                "residual_risk": "Monitor payments latency for next 1h window",
            }

        # Simulate realistic coordination progress - not execution simulation
        total_steps = len([step for step in steps if isinstance(step, Mapping)])
        # For coordination state, show handoff progress not execution completion
        if status_state == "coordination_active":
            completed_count = min(
                2, total_steps
            )  # Show realistic coordination progress
            completed = [
                step for step in steps[:completed_count] if isinstance(step, Mapping)
            ]
            pending = [
                step for step in steps[completed_count:] if isinstance(step, Mapping)
            ]
            summary = f"Coordinating mitigation plan {plan.get('plan_id')}: {completed_count}/{total_steps} handoffs completed"
        else:
            # Default to showing first step in progress
            completed = [step for step in steps[:1] if isinstance(step, Mapping)]
            pending = [step for step in steps[1:] if isinstance(step, Mapping)]
            summary = (
                f"Coordinating mitigation plan {plan.get('plan_id')}: {completed[0].get('action')} handed off to execution team."
                if completed
                else f"Coordinating mitigation plan {plan.get('plan_id')}"
            )
        return {
            "summary": summary,
            "completed_steps": [step.get("action") for step in completed],
            "pending_steps": [step.get("action") for step in pending],
            "next_update_eta": "15m",
            "communication": {
                "internal": communications.get("internal"),
            },
        }

    def _transform_evidence_item(self, item: Mapping[str, object]) -> EvidenceReference:
        """Transform LLM evidence format to EvidenceReference model format."""
        # Handle different LLM response formats
        if "title" in item:
            # Already in correct format
            return EvidenceReference(**item)

        # Transform from LLM format: {"type": "...", "reference": "..."}
        evidence_type = item.get("type", "evidence")
        reference = item.get("reference", "")

        # Create a descriptive title from type and reference
        title = f"{evidence_type.replace('_', ' ').title()}"
        if reference:
            title = f"{title}: {reference}"

        return EvidenceReference(
            title=title[:100],  # Limit title length
            href=reference if reference.startswith(('http', 'https')) else None,
            summary=item.get("summary")
        )


__all__ = ["MitigationCommsAgent", "MitigationAgentConfig"]
