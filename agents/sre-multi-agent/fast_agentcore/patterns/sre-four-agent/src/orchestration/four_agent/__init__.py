"""Foundational components and convenience exports for the four-agent demo."""

from __future__ import annotations

from . import scenario_loader, schema, state, transcript
from .analyst_agent import AnalystAgent, AnalystAgentConfig

# NOTE: approval_agent module not found - commenting out for pipeline compatibility
# from .approval_agent import (
#     ApprovalAgentConfig,
#     DecisionScript,
#     LLMApprovalAgent,
#     ScriptedApprovalAgent,
# )
# Cache and console UI removed for simplified demo
from .demo import (
    available_scenarios,
    DEFAULT_CSV_SCENARIOS,
    DEFAULT_SCENARIOS,
    DemoRun,
    KAFKA_SCENARIOS_AVAILABLE,
    run_all_scenarios,
    run_demo_scenario,
)
from .impact_agent import ImpactAgent, ImpactAgentConfig
from .interfaces import AsyncCallableAgent, ConversationAgent, ensure_agent
from .mitigation_agent import MitigationAgentConfig, MitigationCommsAgent
from .orchestrator import PhaseTwoOrchestrator, PhaseTwoResult
from .rca_agent import Hypothesis, RCAAgent
from .scenario_loader import (
    load_snapshot,
    load_snapshots,
    SCENARIO_REGISTRY,
    ScenarioMetadata,
    ScenarioSnapshot,
)
from .schema import (
    AgentMessage,
    AgentRole,
    ApprovalDecision,
    EvidenceReference,
    MessageType,
    PayloadModel,
    Severity,
    TranscriptEntry,
)

# SignalsAgent removed - replaced with AnalystAgent for simplified workflow
from .summary import SummaryExporter

# ApprovalRequirement = state.ApprovalRequirement  # Doesn't exist in state module
IncidentState = state.IncidentState
TranscriptLogger = transcript.TranscriptLogger
read_transcript = transcript.read_transcript

__all__ = [
    "schema",
    "transcript",
    "state",
    "scenario_loader",
    "AgentRole",
    "MessageType",
    "Severity",
    "EvidenceReference",
    "PayloadModel",
    "AgentMessage",
    "ApprovalDecision",
    "TranscriptEntry",
    # "ApprovalRequirement",  # Doesn't exist in state module
    "IncidentState",
    "ScenarioMetadata",
    "ScenarioSnapshot",
    "SCENARIO_REGISTRY",
    "load_snapshot",
    "load_snapshots",
    "TranscriptLogger",
    "read_transcript",
    "AnalystAgent",
    "AnalystAgentConfig",
    # "SignalsAgent", "SignalsAgentConfig" - removed for simplified workflow
    "RCAAgent",
    "Hypothesis",
    "ImpactAgent",
    "ImpactAgentConfig",
    "MitigationCommsAgent",
    "MitigationAgentConfig",
    "PhaseTwoOrchestrator",
    "PhaseTwoResult",
    "ConversationAgent",
    "AsyncCallableAgent",
    "ensure_agent",
    # "ScriptedApprovalAgent",  # Commented out - approval_agent module missing
    # "DecisionScript",
    # "LLMApprovalAgent",
    # "ApprovalAgentConfig",
    # Console UI components removed for simplified demo
    "SummaryExporter",
    "DemoRun",
    "run_demo_scenario",
    "run_all_scenarios",
    "available_scenarios",
    "DEFAULT_SCENARIOS",
    "DEFAULT_CSV_SCENARIOS",
    "KAFKA_SCENARIOS_AVAILABLE",
]
