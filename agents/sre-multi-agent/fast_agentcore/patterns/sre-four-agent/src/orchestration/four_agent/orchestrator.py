"""LangGraph-based orchestrator for the four-agent incident response demo."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Tuple

from langgraph.errors import GraphInterrupt, Interrupt
from langgraph.graph import END, START, StateGraph

from ..observability import emit_event, wrap_payload
from .impact_agent import ImpactAgent
from .interfaces import ConversationAgent, ensure_agent
from .mitigation_agent import MitigationCommsAgent
from .rca_agent import RCAAgent
from .scenario_loader import ScenarioSnapshot
from .schema import (
    AgentMessage,
    AgentRole,
    MessageType,
    PayloadModel,
    Severity,
)
from .analyst_agent import AnalystAgent
from .state import IncidentState
from .summary import IncidentSummary, SummaryExporter, build_incident_summary
from .transcript import TranscriptLogger


class OrchestrationError(Exception):
    """Exception raised when orchestration encounters unrecoverable errors."""

    pass


@dataclass
class PhaseTwoResult:
    """Aggregate artefacts produced by the LangGraph orchestrator."""

    state: IncidentState
    requests: List[AgentMessage] = field(default_factory=list)
    responses: List[AgentMessage] = field(default_factory=list)
    plans: List[AgentMessage] = field(default_factory=list)
    # Removed approvals field as part of simplification
    status_updates: List[AgentMessage] = field(default_factory=list)
    summary: Optional[IncidentSummary] = None


@dataclass
class _GraphRuntimeState:
    """Mutable state shared between LangGraph nodes."""

    snapshot: ScenarioSnapshot
    result: PhaseTwoResult
    base_context: Dict[str, object]
    current_plan: Optional[AgentMessage] = None


class PhaseTwoOrchestrator:
    """LangGraph-driven orchestration for the incident response workflow."""

    def __init__(
        self,
        analyst_agent: AnalystAgent | ConversationAgent,
        rca_agent: RCAAgent | ConversationAgent,
        impact_agent: ImpactAgent | ConversationAgent,
        mitigation_agent: MitigationCommsAgent | ConversationAgent,
        *,
        transcript_logger: TranscriptLogger | None = None,
        summary_exporter: SummaryExporter | None = None,
        demo_mode: bool = False,
        incident_id: Optional[str] = None,
        resource_pool=None,
    ) -> None:
        self._analyst = ensure_agent(analyst_agent)
        self._rca = ensure_agent(rca_agent)
        self._impact = ensure_agent(impact_agent)
        self._mitigation = ensure_agent(mitigation_agent)
        self._logger = transcript_logger
        self._summary_exporter = summary_exporter
        self._demo_mode = demo_mode

        # Enhanced for concurrent incident support
        self.incident_id = incident_id  # Unique identifier for state isolation
        self.resource_pool = resource_pool  # Shared resource access

        # State tracking for WebSocket event emission
        self._current_stage: Optional[str] = None
        self._execution_timeline: List[dict] = []
        self._lock = asyncio.Lock()

        self._graph = self._build_graph()
        self._workflow = self._graph.compile()
        self._pending_state: _GraphRuntimeState | None = None
        self._pending_interrupt: GraphInterrupt | None = None

    # ------------------------------------------------------------------
    # State tracking methods for WebSocket event emission
    # ------------------------------------------------------------------
    def get_current_stage(self) -> Optional[str]:
        """Get which agent is currently executing."""
        return self._current_stage

    def get_execution_timeline(self) -> List[dict]:
        """Get timeline of agent executions."""
        return self._execution_timeline.copy()

    def _set_stage(self, stage: str) -> None:
        """Internal: Update current stage."""
        self._current_stage = stage
        self._execution_timeline.append(
            {
                "stage": stage,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "stage_change",
            }
        )
        print(f"🎯 Stage set to: {stage}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def run(
        self, snapshot: ScenarioSnapshot, *, auto_resume: bool = True
    ) -> PhaseTwoResult:
        """Execute the incident workflow for *snapshot*."""

        if self._pending_state is not None:
            raise RuntimeError("Cannot start a new run while another is suspended")

        _ = auto_resume  # Reserved for future auto-resume strategies
        runtime = self._initial_runtime(snapshot)
        reset = getattr(self._workflow, "reset", None)
        if callable(reset):
            reset()
        else:
            self._workflow = self._graph.compile()

        final_runtime = await self._drive(runtime)
        return final_runtime.result

    async def resume(
        self,
        approval_message: AgentMessage,
        *,
        auto_resume: bool = True,
    ) -> PhaseTwoResult:
        """Resume method kept for API compatibility. No manual approvals in simplified workflow."""

        # In the simplified workflow, there are no approval interrupts
        # This method is kept for backward compatibility but should not be needed
        raise RuntimeError(
            "Resume not supported in simplified workflow. "
            "The streamlined system operates autonomously without approval interrupts."
        )

    def visualize(self) -> str:
        """Return an ASCII representation of the LangGraph workflow."""

        # LangGraph doesn't have draw_ascii, return a simple text representation
        nodes = list(self._graph.nodes.keys())
        edges = [f"{source} -> {target}" for source, target in self._graph.edges]

        return f"Nodes: {nodes}\nEdges: {edges}"

    # ------------------------------------------------------------------
    # Graph construction and execution helpers
    # ------------------------------------------------------------------
    def _build_graph(self) -> StateGraph:
        graph = StateGraph(_GraphRuntimeState)
        graph.add_node("analyst", self._analyst_node)
        graph.add_node("rca", self._rca_node)
        graph.add_node("impact", self._impact_node)
        graph.add_node("mitigation", self._mitigation_node)
        graph.add_node("summary", self._summary_node)

        graph.add_edge(START, "analyst")
        graph.add_edge("analyst", "rca")
        graph.add_edge("rca", "impact")
        graph.add_edge("impact", "mitigation")
        graph.add_edge("mitigation", "summary")
        graph.add_edge("summary", END)
        return graph

    async def _drive(self, runtime: _GraphRuntimeState) -> _GraphRuntimeState:
        """Drive the workflow with circuit breaker protection."""
        # Circuit breaker constants
        MAX_ITERATIONS = 100
        MAX_EXECUTION_TIME = 300  # 5 minutes

        iteration_count = 0
        start_time = time.time()

        while True:
            # Circuit breaker: Check iteration limit
            iteration_count += 1
            if iteration_count > MAX_ITERATIONS:
                raise OrchestrationError(
                    f"Workflow exceeded maximum iterations ({MAX_ITERATIONS}). "
                    "This indicates a potential infinite loop in the orchestration."
                )

            # Circuit breaker: Check timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > MAX_EXECUTION_TIME:
                raise OrchestrationError(
                    f"Workflow exceeded maximum execution time ({MAX_EXECUTION_TIME} seconds). "
                    "This indicates a potential timeout or deadlock in the orchestration."
                )

            try:
                # Add timeout to individual invocation as additional safety
                result = await asyncio.wait_for(
                    self._workflow.ainvoke(runtime),
                    timeout=60.0,  # 1 minute per individual invocation
                )

                # LangGraph returns the final state, which should be our runtime
                if isinstance(result, dict) and "runtime" in result:
                    runtime = result["runtime"]
                elif isinstance(result, dict) and "state" in result:
                    runtime = result["state"]
                elif isinstance(result, _GraphRuntimeState):
                    runtime = result
                else:
                    # If result is a dict, extract the last state passed to nodes
                    runtime = runtime  # Keep the runtime we passed in
            except GraphInterrupt as interrupt:
                self._pending_state = runtime
                self._pending_interrupt = interrupt
                raise
            except asyncio.TimeoutError:
                raise OrchestrationError(
                    "Individual workflow invocation timed out (60 seconds). "
                    "This indicates an unresponsive agent or LLM service."
                )
            else:
                self._pending_state = None
                self._pending_interrupt = None
                return runtime

    def _initial_runtime(self, snapshot: ScenarioSnapshot) -> _GraphRuntimeState:
        incident_state = self._initial_incident_state(snapshot)
        result = PhaseTwoResult(state=incident_state)
        base_context = {
            "monitoring": snapshot.monitoring,
            "additional_sources": snapshot.additional_sources,
            "demo_mode": self._demo_mode,
        }
        return _GraphRuntimeState(
            snapshot=snapshot,
            result=result,
            base_context=base_context,
        )

    def _initial_incident_state(self, snapshot: ScenarioSnapshot) -> IncidentState:
        state = IncidentState(
            incident_id=snapshot.incident_id,
            severity=snapshot.metadata.severity,
        )
        state.metadata["scenario_key"] = snapshot.metadata.key
        state.metadata["description"] = snapshot.metadata.description
        state.metadata["window_minutes"] = round(
            (snapshot.window.end - snapshot.window.start).total_seconds() / 60.0,
            2,
        )
        state.metadata["demo_mode"] = self._demo_mode
        return state

    # ------------------------------------------------------------------
    # LangGraph node implementations
    # ------------------------------------------------------------------
    async def _analyst_node(
        self, *args: object, **kwargs: object
    ) -> _GraphRuntimeState:
        self._set_stage("analyst")  # Track stage for WebSocket events
        runtime = self._runtime_from_args(args, kwargs)
        request = self._build_analysis_request("analyst", runtime)
        self._record_message(
            runtime, request, runtime.result.requests, "analyst", "request"
        )
        self._emit_stage_start(runtime, "analyst")
        response, metadata = await self._invoke_agent(
            "analyst", self._analyst, request, runtime
        )
        if response is not None:
            self._record_message(
                runtime,
                response,
                runtime.result.responses,
                "analyst",
                "response",
                extra_metadata=metadata,
            )
        self._emit_stage_completed(runtime, "analyst", metadata)
        return runtime

    async def _rca_node(self, *args: object, **kwargs: object) -> _GraphRuntimeState:
        self._set_stage("rca")  # Track stage for WebSocket events
        runtime = self._runtime_from_args(args, kwargs)
        request = self._build_analysis_request("rca", runtime)
        self._record_message(
            runtime, request, runtime.result.requests, "rca", "request"
        )
        self._emit_stage_start(runtime, "rca")
        response, metadata = await self._invoke_agent(
            "rca", self._rca, request, runtime
        )
        if response is not None:
            self._record_message(
                runtime,
                response,
                runtime.result.responses,
                "rca",
                "response",
                extra_metadata=metadata,
            )
        self._emit_stage_completed(runtime, "rca", metadata)
        return runtime

    async def _impact_node(self, *args: object, **kwargs: object) -> _GraphRuntimeState:
        self._set_stage("impact")  # Track stage for WebSocket events
        runtime = self._runtime_from_args(args, kwargs)
        request = self._build_analysis_request("impact", runtime)
        self._record_message(
            runtime, request, runtime.result.requests, "impact", "request"
        )
        self._emit_stage_start(runtime, "impact")
        response, metadata = await self._invoke_agent(
            "impact", self._impact, request, runtime
        )
        if response is not None:
            self._record_message(
                runtime,
                response,
                runtime.result.responses,
                "impact",
                "response",
                extra_metadata=metadata,
            )
        self._emit_stage_completed(runtime, "impact", metadata)
        return runtime

    async def _mitigation_node(
        self, *args: object, **kwargs: object
    ) -> _GraphRuntimeState:
        self._set_stage("mitigation")  # Track stage for WebSocket events
        runtime = self._runtime_from_args(args, kwargs)
        stage = "mitigation_plan"
        revision = len(runtime.result.plans)
        request = self._build_mitigation_request(runtime, revision=revision)
        self._record_message(
            runtime, request, runtime.result.requests, stage, "request"
        )
        self._emit_stage_start(runtime, stage, revision=revision)
        response, metadata = await self._invoke_agent(
            stage, self._mitigation, request, runtime
        )
        if response is None:
            return runtime

        self._record_message(
            runtime,
            response,
            runtime.result.responses,
            stage,
            "response",
            extra_metadata=metadata,
        )
        runtime.result.plans.append(response)
        runtime.current_plan = response
        plan_id = response.payload.details.get("plan_id", "plan")
        self._emit_stage_completed(runtime, stage, metadata, plan_id=plan_id)
        return runtime

    async def _summary_node(
        self, *args: object, **kwargs: object
    ) -> _GraphRuntimeState:
        runtime = self._runtime_from_args(args, kwargs)
        summary = build_incident_summary(
            runtime.result.state,
            runtime.result.responses,
            runtime.result.plans,
            [],  # No approvals in simplified workflow
            runtime.result.status_updates,
            runtime.snapshot,
        )
        runtime.result.summary = summary
        if self._summary_exporter is not None:
            path = self._summary_exporter.write(summary)
            emit_event(
                "orchestrator",
                "summary_exported",
                wrap_payload(
                    incident_id=runtime.result.state.incident_id,
                    severity=runtime.result.state.severity.value,
                    path=str(path),
                ),
            )
        return runtime

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _runtime_from_args(
        self,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> _GraphRuntimeState:
        """Extract _GraphRuntimeState from LangGraph node arguments with type safety.

        LangGraph passes runtime state through various parameter patterns:
        - As kwarg: runtime=state, state=state, input=state
        - As positional arg: first argument

        Args:
            args: Positional arguments from LangGraph node call
            kwargs: Keyword arguments from LangGraph node call

        Returns:
            The extracted _GraphRuntimeState instance

        Raises:
            TypeError: If no runtime found or runtime has incorrect type
        """
        runtime_candidate: object = None

        # Check for runtime in kwargs with preferred key order
        runtime_keys = ("runtime", "state", "input")
        for key in runtime_keys:
            if key in kwargs:
                runtime_candidate = kwargs[key]
                break

        # Fallback to first positional argument if no kwarg found
        if runtime_candidate is None and args:
            runtime_candidate = args[0]

        # Type guard: ensure we found a runtime candidate
        if runtime_candidate is None:
            available_keys = list(kwargs.keys()) if kwargs else []
            args_count = len(args)
            raise TypeError(
                f"No runtime argument found. Expected one of {runtime_keys} in kwargs "
                f"or runtime as first positional arg. "
                f"Got kwargs keys: {available_keys}, args count: {args_count}"
            )

        # Type guard: ensure runtime candidate is correct type
        if not isinstance(runtime_candidate, _GraphRuntimeState):
            runtime_type = type(runtime_candidate).__name__
            runtime_module = getattr(type(runtime_candidate), "__module__", "unknown")
            raise TypeError(
                f"Expected runtime argument to be _GraphRuntimeState, "
                f"but received {runtime_type} from module {runtime_module}. "
                f"Value: {repr(runtime_candidate)[:100]}"
            )

        # Type narrowing: mypy now knows this is _GraphRuntimeState
        return runtime_candidate

    def _build_analysis_request(
        self, stage: str, runtime: _GraphRuntimeState
    ) -> AgentMessage:
        details: Dict[str, object] = dict(runtime.base_context)
        details["stage"] = stage
        details["prior_messages"] = [
            {
                "from": message.sender.value,
                "type": message.type.value,
                "summary": message.payload.summary,
            }
            for message in runtime.result.responses
        ]

        if stage in {"rca", "impact"} and runtime.result.responses:
            details["analyst"] = runtime.result.responses[0].payload.details
        if stage == "impact" and len(runtime.result.responses) >= 2:
            details["rca"] = runtime.result.responses[1].payload.details

        stage_to_agent = {
            "analyst": AgentRole.ANALYST,  # Using ANALYST alias (resolves to "Signals") as noted in AnalystAgent
            "rca": AgentRole.RCA,
            "impact": AgentRole.IMPACT,
        }
        task_instructions = {
            "analyst": "Analyze unstructured logs and detect anomalies for incident assessment.",
            "rca": "Determine the root cause based on analyst findings.",
            "impact": "Quantify the business impact using agent findings.",
        }

        recipient = stage_to_agent.get(stage)
        summary = task_instructions.get(
            stage,
            f"Execute {stage} stage operations for incident {runtime.result.state.incident_id}",
        )
        message_data = {
            "incident_id": runtime.result.state.incident_id,
            "from": AgentRole.ORCHESTRATOR,
            "type": MessageType.REQUEST,
            "severity": runtime.result.state.severity,
            "payload": PayloadModel(summary=summary, details=details),
        }
        if recipient:
            message_data["to"] = recipient
        return AgentMessage(**message_data)

    def _build_mitigation_request(
        self,
        runtime: _GraphRuntimeState,
        *,
        revision: int,
    ) -> AgentMessage:
        details: Dict[str, object] = dict(runtime.base_context)
        details["stage"] = "mitigation_plan"
        details["intent"] = "plan"
        details["revision"] = revision
        details["prior_messages"] = [
            {
                "from": message.sender.value,
                "type": message.type.value,
                "summary": message.payload.summary,
            }
            for message in runtime.result.responses
        ]
        return AgentMessage(
            incident_id=runtime.result.state.incident_id,
            **{"from": AgentRole.ORCHESTRATOR},
            **{"to": AgentRole.MITIGATION},
            type=MessageType.REQUEST,
            severity=runtime.result.state.severity,
            payload=PayloadModel(
                summary="Create comprehensive incident mitigation plan with recommended actions",
                details=details,
            ),
        )

    async def _invoke_agent(
        self,
        stage: str,
        agent: ConversationAgent,
        request: AgentMessage,
        runtime: _GraphRuntimeState,
    ) -> Tuple[Optional[AgentMessage], Dict[str, object]]:
        start = time.perf_counter()
        response = await agent.handle(request, runtime.result.state)
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
        metadata = {
            "elapsed_ms": elapsed_ms,
            "cache_hit": False,
            "attempts": 1,
        }
        return response, metadata

    def _record_message(
        self,
        runtime: _GraphRuntimeState,
        message: AgentMessage,
        collection: List[AgentMessage],
        stage: str,
        direction: str,
        *,
        extra_metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        runtime.result.state.add_message(message)
        collection.append(message)
        if self._logger is not None:
            metadata: Dict[str, object] = {"stage": stage, "direction": direction}
            if extra_metadata:
                for k, v in extra_metadata.items():
                    if v is not None:
                        metadata[k] = v
            self._logger.append(message, metadata=metadata)

    def _emit_stage_start(
        self,
        runtime: _GraphRuntimeState,
        stage: str,
        **extra: object,
    ) -> None:
        emit_event(
            "orchestrator",
            "stage_start",
            wrap_payload(
                incident_id=runtime.result.state.incident_id,
                severity=runtime.result.state.severity.value,
                stage=stage,
                **extra,
            ),
        )

    def _emit_stage_completed(
        self,
        runtime: _GraphRuntimeState,
        stage: str,
        metadata: Mapping[str, object],
        **extra: object,
    ) -> None:
        emit_event(
            "orchestrator",
            "stage_completed",
            wrap_payload(
                incident_id=runtime.result.state.incident_id,
                severity=runtime.result.state.severity.value,
                stage=stage,
                duration_ms=metadata.get("elapsed_ms"),
                attempts=metadata.get("attempts"),
                cache_hit=metadata.get("cache_hit"),
                **extra,
            ),
        )


__all__ = ["PhaseTwoOrchestrator", "PhaseTwoResult"]
