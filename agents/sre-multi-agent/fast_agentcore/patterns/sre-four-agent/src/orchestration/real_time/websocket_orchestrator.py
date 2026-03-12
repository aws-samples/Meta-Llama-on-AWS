"""
WebSocket-enabled orchestrator wrapper for real-time pipeline updates.

This module wraps the existing PhaseTwoOrchestrator to emit real-time
WebSocket updates during pipeline execution.
"""

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

from ..four_agent.analyst_agent import AnalystAgent
from ..four_agent.impact_agent import ImpactAgent
from ..four_agent.mitigation_agent import MitigationCommsAgent
from ..four_agent.orchestrator import PhaseTwoOrchestrator, PhaseTwoResult
from ..four_agent.rca_agent import RCAAgent
from ..four_agent.scenario_loader import ScenarioSnapshot
from ..four_agent.summary import SummaryExporter
from ..four_agent.transcript import TranscriptLogger
from .pipeline_state_manager import AgentStatus, get_pipeline_state_manager


class WebSocketOrchestrator:
    """
    WebSocket-enabled orchestrator that provides real-time updates.

    This wraps the existing PhaseTwoOrchestrator and emits WebSocket events
    for pipeline and agent state changes.
    """

    def __init__(
        self,
        analyst_agent: AnalystAgent,
        rca_agent: RCAAgent,
        impact_agent: ImpactAgent,
        mitigation_agent: MitigationCommsAgent,
        *,
        state_manager: Any | None = None,
        pipeline_id: str | None = None,
        transcript_logger: TranscriptLogger | None = None,
        summary_exporter: SummaryExporter | None = None,
        demo_mode: bool = False,
        incident_id: str | None = None,
    ):
        # Create the underlying orchestrator
        self.orchestrator = PhaseTwoOrchestrator(
            analyst_agent=analyst_agent,
            rca_agent=rca_agent,
            impact_agent=impact_agent,
            mitigation_agent=mitigation_agent,
            transcript_logger=transcript_logger,
            summary_exporter=summary_exporter,
            demo_mode=demo_mode,
            incident_id=incident_id,
        )

        # Use provided state_manager or fall back to singleton
        self.pipeline_manager = (
            state_manager if state_manager is not None else get_pipeline_state_manager()
        )
        self.pipeline_id: str | None = pipeline_id
        self.current_snapshot: ScenarioSnapshot | None = (
            None  # Store snapshot for log access
        )

        print("🔧 WebSocketOrchestrator initialized:")
        print(f"   Pipeline ID: {self.pipeline_id}")
        print(f"   State Manager: {type(self.pipeline_manager).__name__}")
        print(f"   State Manager ID: {id(self.pipeline_manager)}")
        print(
            f"   WebSocket Connections: {len(self.pipeline_manager.websocket_connections) if hasattr(self.pipeline_manager, 'websocket_connections') else 'N/A'}"
        )
        print(
            f"   WebSocket Connections Set ID: {id(self.pipeline_manager.websocket_connections)}"
        )
        self._agent_name_mapping = {
            "analyst": "Initial Analysis",
            "rca": "Root Cause",
            "impact": "Impact Assessment",
            "mitigation": "Solution Planning",
        }

    async def run_with_websocket_updates(
        self, snapshot: ScenarioSnapshot, pipeline_id: str | None = None
    ) -> PhaseTwoResult:
        """
        Execute the pipeline with real-time WebSocket updates.

        Args:
            snapshot: The scenario snapshot to process
            pipeline_id: Optional pipeline ID (will generate one if not provided)

        Returns:
            PhaseTwoResult from the underlying orchestrator
        """
        # Generate pipeline ID if not provided
        if pipeline_id is None:
            pipeline_id = f"pipeline-{int(time.time())}"

        self.pipeline_id = pipeline_id
        self.current_snapshot = snapshot  # Store for log access

        try:
            # Create pipeline state
            await self.pipeline_manager.create_pipeline(
                pipeline_id=pipeline_id,
                incident_id=snapshot.metadata.key,
                severity=snapshot.metadata.severity,
            )

            # Add initial log data - extract from snapshot if available
            log_data: list[dict] = []
            if hasattr(snapshot, "logs") and snapshot.logs:
                # Take first 20 logs as sample
                log_data = snapshot.logs[:20] if isinstance(snapshot.logs, list) else []
                print(f"📋 Extracted {len(log_data)} logs from snapshot")

            await self.pipeline_manager.add_processed_logs(pipeline_id, log_data)

            # Start pipeline
            await self.pipeline_manager.start_pipeline(pipeline_id)
            print(f"📊 Pipeline {pipeline_id} started, running orchestrator...")

            # Start state polling task for real-time event emission
            polling_task = asyncio.create_task(self._poll_orchestrator_state())

            try:
                # Run the base orchestrator (this blocks until complete)
                result = await self.orchestrator.run(snapshot)
                print(f"✅ Orchestrator finished for pipeline {pipeline_id}")
            finally:
                # Stop polling
                polling_task.cancel()
                try:
                    await polling_task
                except asyncio.CancelledError:
                    pass

            # Mark pipeline as complete
            await self.pipeline_manager.complete_pipeline(pipeline_id, success=True)

            return result

        except Exception as e:
            # Mark pipeline as failed
            if self.pipeline_id:
                await self.pipeline_manager.complete_pipeline(
                    pipeline_id, success=False, error_message=str(e)
                )
            raise

    async def _run_with_monitoring(self, snapshot: ScenarioSnapshot) -> PhaseTwoResult:
        """Run the orchestrator with agent monitoring."""

        # Monkey-patch the orchestrator's agent nodes to emit events
        original_analyst_node = self.orchestrator._analyst_node
        original_rca_node = self.orchestrator._rca_node
        original_impact_node = self.orchestrator._impact_node
        original_mitigation_node = self.orchestrator._mitigation_node

        async def monitored_analyst_node(*args, **kwargs):
            await self._emit_agent_start("analyst")
            result = await original_analyst_node(*args, **kwargs)
            await self._emit_agent_complete("analyst", result)
            return result

        async def monitored_rca_node(*args, **kwargs):
            await self._emit_agent_start("rca")
            result = await original_rca_node(*args, **kwargs)
            await self._emit_agent_complete("rca", result)
            return result

        async def monitored_impact_node(*args, **kwargs):
            await self._emit_agent_start("impact")
            result = await original_impact_node(*args, **kwargs)
            await self._emit_agent_complete("impact", result)
            return result

        async def monitored_mitigation_node(*args, **kwargs):
            await self._emit_agent_start("mitigation")
            result = await original_mitigation_node(*args, **kwargs)
            await self._emit_agent_complete("mitigation", result)
            return result

        # Apply monitoring wrapper
        self.orchestrator._analyst_node = monitored_analyst_node
        self.orchestrator._rca_node = monitored_rca_node
        self.orchestrator._impact_node = monitored_impact_node
        self.orchestrator._mitigation_node = monitored_mitigation_node

        try:
            # Run the orchestrator
            return await self.orchestrator.run(snapshot)
        finally:
            # Restore original methods
            self.orchestrator._analyst_node = original_analyst_node
            self.orchestrator._rca_node = original_rca_node
            self.orchestrator._impact_node = original_impact_node
            self.orchestrator._mitigation_node = original_mitigation_node

    async def broadcast_agent_message(
        self,
        icon: str,
        agent: str,
        title: str,
        message: str,
        details: list[str] = None,
        level: str = "info",
    ):
        """Broadcast agent message to WebSocket for UI display."""
        try:
            await self.pipeline_manager.broadcast_update(
                {
                    "type": "agent_stream",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "icon": icon,
                    "level": level,
                    "agent": agent,
                    "title": title,
                    "message": message,
                    "details": details or [],
                }
            )
        except Exception as e:
            # Don't fail the pipeline if broadcast fails
            print(f"Warning: Failed to broadcast agent message: {e}")

    async def _emit_agent_start(self, agent_key: str) -> None:
        """Emit agent start event - Enhanced for Phase 2."""
        if not self.pipeline_id:
            print(f"⚠️  No pipeline_id set for agent {agent_key}")
            return

        agent_name = self._agent_name_mapping.get(agent_key, agent_key)
        print(
            f"🚀 Emitting agent start for {agent_name} (pipeline: {self.pipeline_id})"
        )

        # BROADCAST to WebSocket clients
        await self.pipeline_manager.broadcast_update(
            {
                "type": "agent_started",
                "pipeline_id": self.pipeline_id,
                "agent": agent_key,
                "agent_name": agent_name,
                "message": f"Starting {agent_name} analysis...",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        # Also broadcast to agent stream
        await self.broadcast_agent_message(
            icon=(
                "🔍"
                if agent_key == "analyst"
                else (
                    "📊"
                    if agent_key == "impact"
                    else "🔍" if agent_key == "rca" else "🛠️"
                )
            ),
            agent=agent_key,
            title=f"Orchestrator → {agent_name}",
            message=f"Starting {agent_name.lower()} analysis...",
        )

        # Update internal state
        await self.pipeline_manager.update_agent_processing_stage(
            pipeline_id=self.pipeline_id, agent_name=agent_name, stage="initializing"
        )

        await self.pipeline_manager.update_agent_status(
            pipeline_id=self.pipeline_id,
            agent_name=agent_name,
            status=AgentStatus.PROCESSING,
            progress=0.1,
            message=f"Starting {agent_key} analysis...",
        )

        await self.pipeline_manager.add_agent_activity(
            pipeline_id=self.pipeline_id,
            agent_name=agent_name,
            activity_type="analyzing",
            description=f"Initializing {agent_key} agent with AWS Bedrock",
            details={"timestamp": datetime.now(UTC).isoformat()},
        )

    async def _emit_agent_complete(self, agent_key: str, result) -> None:
        """Emit agent completion event - Enhanced for Phase 2."""
        if not self.pipeline_id:
            print(f"⚠️  No pipeline_id set for agent completion {agent_key}")
            return

        agent_name = self._agent_name_mapping.get(agent_key, agent_key)
        print(
            f"✅ Emitting agent complete for {agent_name} (pipeline: {self.pipeline_id})"
        )

        # Update processing stage
        await self.pipeline_manager.update_agent_processing_stage(
            pipeline_id=self.pipeline_id, agent_name=agent_name, stage="finalizing"
        )

        # Extract findings from the agent result
        findings = []
        confidence = None
        full_output = ""
        hypotheses = []  # For RCA agent

        if hasattr(result, "result") and hasattr(result.result, "responses"):
            # Look for the latest response from this agent
            for response in result.result.responses:
                if response.payload and response.payload.summary:
                    summary = response.payload.summary
                    findings.append(
                        summary[:100] + "..." if len(summary) > 100 else summary
                    )
                    full_output += f"## Analysis Summary\n{summary}\n\n"

                if hasattr(response.payload, "details") and response.payload.details:
                    details = response.payload.details

                    # Special handling for RCA agent - extract ranked hypotheses
                    if agent_key == "rca" and isinstance(details, dict):
                        ranked_hypotheses = details.get("ranked_hypotheses", [])
                        if ranked_hypotheses:
                            full_output += "## Root Cause Hypotheses\n\n"
                            for i, hyp in enumerate(ranked_hypotheses, 1):
                                hypothesis_text = hyp.get("hypothesis", "Unknown")
                                hyp_confidence = hyp.get("confidence", 0.0)
                                evidence = hyp.get("evidence", {})
                                actions = hyp.get("recommended_actions", [])

                                # Add to findings for timeline
                                findings.append(
                                    f"#{i}: {hypothesis_text} ({hyp_confidence:.0%} confidence)"
                                )

                                # Build detailed output
                                full_output += f"### Hypothesis {i} ({hyp_confidence:.0%} confidence)\n"
                                full_output += f"**{hypothesis_text}**\n\n"
                                if evidence:
                                    full_output += f"**Evidence:** {', '.join(f'{k}={v}' for k, v in evidence.items())}\n\n"
                                if actions:
                                    full_output += "**Recommended Actions:**\n"
                                    for action in actions:
                                        full_output += f"- {action}\n"
                                    full_output += "\n"

                                # Store for websocket broadcast
                                hypotheses.append(
                                    {
                                        "hypothesis": hypothesis_text,
                                        "confidence": hyp_confidence,
                                        "evidence": evidence,
                                        "recommended_actions": actions,
                                    }
                                )

                            # Use highest confidence as overall confidence
                            if ranked_hypotheses:
                                confidence = ranked_hypotheses[0].get("confidence", 0.8)
                    else:
                        # Generic handling for other agents
                        confidence = details.get("confidence", 0.8)
                        details_str = str(details)
                        full_output += f"## Detailed Analysis\n{details_str}\n\n"

        # Stream the full output in chunks for better UX
        if full_output:
            await self._stream_agent_output_in_chunks(agent_name, full_output)

        # BROADCAST to WebSocket clients
        broadcast_data = {
            "type": "agent_completed",
            "pipeline_id": self.pipeline_id,
            "agent": agent_key,
            "agent_name": agent_name,
            "message": f"{agent_name} completed analysis",
            "findings": findings,
            "confidence": confidence,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Add RCA-specific data if available
        if agent_key == "rca" and hypotheses:
            broadcast_data["hypotheses"] = hypotheses

        await self.pipeline_manager.broadcast_update(broadcast_data)

        await self.pipeline_manager.update_agent_status(
            pipeline_id=self.pipeline_id,
            agent_name=agent_name,
            status=AgentStatus.COMPLETE,
            progress=1.0,
            message=f"{agent_key} analysis completed",
        )

        await self.pipeline_manager.add_agent_activity(
            pipeline_id=self.pipeline_id,
            agent_name=agent_name,
            activity_type="completing",
            description=f"{agent_key} agent completed analysis",
            details={"completion_time": datetime.now(UTC).isoformat()},
        )

        # Add findings
        for finding in findings:
            await self.pipeline_manager.add_agent_finding(
                pipeline_id=self.pipeline_id,
                agent_name=agent_name,
                finding=finding,
                confidence=confidence,
            )

        # Track agent handoff communication for next agent
        await self._track_agent_handoff(agent_key, result, findings)

    async def _stream_agent_output_in_chunks(
        self, agent_name: str, full_output: str
    ) -> None:
        """Stream agent output in chunks for better UX."""
        chunk_size = 200  # characters per chunk
        chunks = [
            full_output[i : i + chunk_size]
            for i in range(0, len(full_output), chunk_size)
        ]

        for i, chunk_content in enumerate(chunks):
            await self.pipeline_manager.stream_agent_output_chunk(
                pipeline_id=self.pipeline_id,
                agent_name=agent_name,
                chunk_content=chunk_content,
                chunk_type="analysis",
                is_complete=(i == len(chunks) - 1),
                total_chunks=len(chunks),
            )
            # Small delay for realistic streaming effect
            await asyncio.sleep(0.1)

    async def _track_agent_handoff(
        self, current_agent_key: str, result: Any, findings: list[str]
    ) -> None:
        """Track communication between agents for flow visualization."""
        if not self.pipeline_id or not findings:
            return

        current_agent_name = self._agent_name_mapping.get(
            current_agent_key, current_agent_key
        )

        # Determine next agent and data type
        agent_flow = {
            "analyst": ("rca", "analysis"),
            "rca": ("impact", "hypothesis"),
            "impact": ("mitigation", "impact"),
            "mitigation": (None, "plan"),
        }

        next_agent_key, data_type = agent_flow.get(current_agent_key, (None, "data"))

        if next_agent_key:
            next_agent_name = self._agent_name_mapping.get(
                next_agent_key, next_agent_key
            )

            # Calculate confidence from result
            confidence = 0.8  # default
            if hasattr(result, "result") and hasattr(result.result, "responses"):
                for response in result.result.responses:
                    if (
                        hasattr(response.payload, "details")
                        and response.payload.details
                    ):
                        confidence = response.payload.details.get("confidence", 0.8)
                        break

            # BROADCAST agent communication
            await self.pipeline_manager.broadcast_update(
                {
                    "type": "agent_communication",
                    "pipeline_id": self.pipeline_id,
                    "from_agent": current_agent_key,
                    "from_agent_name": current_agent_name,
                    "to_agent": next_agent_key,
                    "to_agent_name": next_agent_name,
                    "message": (
                        findings[0][:100]
                        if findings
                        else f"{current_agent_name} completed analysis"
                    ),
                    "data_type": data_type,
                    "confidence": confidence,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

            await self.pipeline_manager.add_agent_communication(
                pipeline_id=self.pipeline_id,
                from_agent=current_agent_name,
                to_agent=next_agent_name,
                data_type=data_type,
                data_summary=(
                    findings[0] if findings else f"{current_agent_name} output"
                ),
                data_size=len(findings),
                confidence_score=confidence,
            )

    def get_pipeline_id(self) -> str | None:
        """Get the current pipeline ID."""
        return self.pipeline_id

    async def run_pipeline(self, snapshot: ScenarioSnapshot) -> PhaseTwoResult:
        """
        Run pipeline with state polling for WebSocket events.
        """
        print("🎯 WebSocketOrchestrator.run_pipeline() called")
        print(f"   Pipeline ID: {self.pipeline_id}")

        try:
            # EMIT: Pipeline started
            await self.pipeline_manager.broadcast_update(
                {
                    "type": "pipeline_started",
                    "pipeline_id": self.pipeline_id,
                    "incident_id": snapshot.metadata.key,
                    "severity": snapshot.metadata.severity.name,
                    "description": snapshot.metadata.description,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

            # Broadcast to agent stream
            await self.broadcast_agent_message(
                icon="🎯",
                agent="orchestrator",
                title="Orchestrator - Starting Pipeline",
                message="Initiating incident response workflow",
                details=[
                    f"Incident ID: {snapshot.metadata.key}",
                    f"Severity: {snapshot.metadata.severity.name}",
                ],
            )

            # Create pipeline state
            if self.pipeline_id:
                await self.pipeline_manager.create_pipeline(
                    pipeline_id=self.pipeline_id,
                    incident_id=snapshot.metadata.key,
                    severity=snapshot.metadata.severity,
                )
                await self.pipeline_manager.start_pipeline(self.pipeline_id)

            # Start state polling task
            polling_task = asyncio.create_task(self._poll_orchestrator_state())

            print(
                f"🔵 Starting base orchestrator.run() for incident {snapshot.metadata.key}"
            )
            print(
                f"🔵 Snapshot metadata: severity={snapshot.metadata.severity}, description={snapshot.metadata.description}"
            )

            try:
                # Run base orchestrator (this blocks until complete)
                result = await self.orchestrator.run(snapshot)
                print("🔵 Base orchestrator.run() completed successfully")
                print(f"🔵 Result type: {type(result)}")
            except Exception as e:
                print(f"❌❌❌ CRITICAL ERROR in orchestrator.run(): {e}")
                print(f"❌ Error type: {type(e).__name__}")
                import traceback

                traceback.print_exc()

                # Broadcast error to UI
                await self.broadcast_agent_message(
                    icon="❌",
                    agent="system",
                    title="Pipeline Error",
                    message=f"Critical error: {str(e)}",
                    level="error",
                )
                raise
            finally:
                # Stop polling
                polling_task.cancel()
                try:
                    await polling_task
                except asyncio.CancelledError:
                    pass

            # EMIT: Pipeline completed
            await self.pipeline_manager.broadcast_update(
                {
                    "type": "pipeline_completed",
                    "pipeline_id": self.pipeline_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

            if self.pipeline_id:
                await self.pipeline_manager.complete_pipeline(self.pipeline_id)

            return result

        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            import traceback

            traceback.print_exc()

            await self.pipeline_manager.broadcast_update(
                {
                    "type": "pipeline_failed",
                    "pipeline_id": self.pipeline_id,
                    "error": str(e),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

            if self.pipeline_id:
                await self.pipeline_manager.complete_pipeline(
                    self.pipeline_id, success=False, error_message=str(e)
                )

            raise

    async def _poll_orchestrator_state(self):
        """
        Poll orchestrator state and emit WebSocket events when stage changes.
        Enhanced with log data for Phase 2.
        """
        last_stage = None

        print("🔍 Starting state polling...")

        while True:
            try:
                await asyncio.sleep(0.5)  # Poll every 500ms

                # Check current stage
                current_stage = self.orchestrator.get_current_stage()

                if current_stage and current_stage != last_stage:
                    print(f"📡 Stage changed: {last_stage} → {current_stage}")

                    # Debug: Check WebSocket connections before broadcasting
                    ws_count = (
                        len(self.pipeline_manager.websocket_connections)
                        if hasattr(self.pipeline_manager, "websocket_connections")
                        else 0
                    )
                    print(f"🔍 DEBUG: WebSocket connections available: {ws_count}")
                    print(f"🔍 DEBUG: pipeline_manager ID: {id(self.pipeline_manager)}")
                    print(
                        f"🔍 DEBUG: websocket_connections set: {self.pipeline_manager.websocket_connections}"
                    )

                    # Extract log sample for this agent
                    log_sample = self._get_log_sample_for_agent(current_stage)

                    agent_name = self._agent_name_mapping.get(
                        current_stage, current_stage
                    )

                    # Emit agent_started with log data
                    event_data = {
                        "type": "agent_started",
                        "pipeline_id": self.pipeline_id,
                        "agent": current_stage,
                        "agent_name": agent_name,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }

                    # Add logs if available
                    if log_sample:
                        event_data["logs"] = log_sample
                        print(
                            f"📋 Including {len(log_sample)} log entries in agent_started event"
                        )

                    await self.pipeline_manager.broadcast_update(event_data)

                    # ALSO broadcast to agent stream
                    icon = (
                        "🔍"
                        if current_stage == "analyst"
                        else (
                            "📊"
                            if current_stage == "impact"
                            else "🔍" if current_stage == "rca" else "🛠️"
                        )
                    )
                    await self.broadcast_agent_message(
                        icon=icon,
                        agent=current_stage,
                        title=f"Orchestrator → {agent_name}",
                        message=f"Starting {agent_name.lower()} analysis...",
                    )
                    print(f"✅ Broadcasted agent_stream message for {agent_name}")

                    # If last_stage exists, emit completion for it
                    if last_stage:
                        await self.pipeline_manager.broadcast_update(
                            {
                                "type": "agent_completed",
                                "pipeline_id": self.pipeline_id,
                                "agent": last_stage,
                                "agent_name": self._agent_name_mapping.get(
                                    last_stage, last_stage
                                ),
                                "timestamp": datetime.now(UTC).isoformat(),
                            }
                        )

                        # Emit communication/handoff
                        await self.pipeline_manager.broadcast_update(
                            {
                                "type": "agent_communication",
                                "pipeline_id": self.pipeline_id,
                                "from_agent": last_stage,
                                "to_agent": current_stage,
                                "message": f"{self._agent_name_mapping.get(last_stage, last_stage)} completed, handing off to {self._agent_name_mapping.get(current_stage, current_stage)}",
                                "timestamp": datetime.now(UTC).isoformat(),
                            }
                        )

                    last_stage = current_stage

            except asyncio.CancelledError:
                print("🛑 Polling cancelled")

                # Emit completion for last stage if exists
                if last_stage:
                    await self.pipeline_manager.broadcast_update(
                        {
                            "type": "agent_completed",
                            "pipeline_id": self.pipeline_id,
                            "agent": last_stage,
                            "agent_name": self._agent_name_mapping.get(
                                last_stage, last_stage
                            ),
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )

                raise
            except Exception as e:
                print(f"❌ Polling error: {e}")
                import traceback

                traceback.print_exc()

    def _get_log_sample_for_agent(
        self, agent_key: str, sample_size: int = 10
    ) -> list[dict]:
        """
        Extract a sample of logs relevant for the current agent.

        Args:
            agent_key: The agent requesting logs (analyst, rca, impact, mitigation)
            sample_size: Number of log entries to return

        Returns:
            List of log dictionaries
        """
        if not self.current_snapshot or not hasattr(self.current_snapshot, "logs"):
            return []

        logs = self.current_snapshot.logs
        if not logs or not isinstance(logs, list):
            return []

        # For analyst, return first logs (initial analysis)
        if agent_key == "analyst":
            return logs[:sample_size]

        # For other agents, return a different slice to show progression
        offset_map = {
            "rca": sample_size,
            "impact": sample_size * 2,
            "mitigation": sample_size * 3,
        }

        offset = offset_map.get(agent_key, 0)
        return logs[offset : offset + sample_size]

    async def _run_agents_with_broadcasts(
        self, snapshot: ScenarioSnapshot
    ) -> PhaseTwoResult:
        """
        Run the base orchestrator and emit WebSocket events.

        The base orchestrator handles the LangGraph workflow properly.
        We just intercept to emit WebSocket events.
        """
        print("🏃 Running base orchestrator with event interception...")

        # Monkey-patch the base orchestrator's agent nodes to emit events
        original_analyst_node = self.orchestrator._analyst_node
        original_rca_node = self.orchestrator._rca_node
        original_impact_node = self.orchestrator._impact_node
        original_mitigation_node = self.orchestrator._mitigation_node

        async def wrapped_analyst(state):
            await self._emit_agent_start("analyst")
            result = await original_analyst_node(state)
            # Note: Don't pass result to _emit_agent_complete - AnalysisResult doesn't have recommended_action anymore
            await self._emit_agent_complete("analyst", None)
            await self._track_agent_handoff(
                "analyst", state, ["Analyst analysis complete"]
            )
            return result

        async def wrapped_rca(state):
            await self._emit_agent_start("rca")
            result = await original_rca_node(state)
            await self._emit_agent_complete("rca", result)
            await self._track_agent_handoff("rca", result, ["RCA complete"])
            return result

        async def wrapped_impact(state):
            await self._emit_agent_start("impact")
            result = await original_impact_node(state)
            await self._emit_agent_complete("impact", result)
            await self._track_agent_handoff(
                "impact", result, ["Impact assessment complete"]
            )
            return result

        async def wrapped_mitigation(state):
            await self._emit_agent_start("mitigation")
            result = await original_mitigation_node(state)
            await self._emit_agent_complete("mitigation", result)
            await self._track_agent_handoff(
                "mitigation", result, ["Mitigation plan complete"]
            )
            return result

        # Replace nodes temporarily
        self.orchestrator._analyst_node = wrapped_analyst
        self.orchestrator._rca_node = wrapped_rca
        self.orchestrator._impact_node = wrapped_impact
        self.orchestrator._mitigation_node = wrapped_mitigation

        try:
            # Run the base orchestrator
            result = await self.orchestrator.run(snapshot)
            return result
        finally:
            # Restore original nodes
            self.orchestrator._analyst_node = original_analyst_node
            self.orchestrator._rca_node = original_rca_node
            self.orchestrator._impact_node = original_impact_node
            self.orchestrator._mitigation_node = original_mitigation_node
