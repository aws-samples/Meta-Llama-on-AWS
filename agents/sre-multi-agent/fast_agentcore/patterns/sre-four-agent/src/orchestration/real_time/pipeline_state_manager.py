"""
Real-time pipeline state management for WebSocket integration.

This module provides centralized state tracking for the incident response pipeline,
enabling real-time updates to be sent to connected WebSocket clients.

Phase 2 enhancements:
- Agent communication tracking for flow visualization
- Real-time output streaming with chunked responses
- Enhanced WebSocket events for detailed agent communication
- Multi-pipeline support with concurrent execution
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

from ..four_agent.schema import AgentRole, Severity


class PipelineStatus(Enum):
    """Pipeline execution status."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentStatus(Enum):
    """Individual agent status."""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"
    PAUSED = "paused"  # Phase 2: Pipeline control


@dataclass
class AgentCommunication:
    """Tracks communication between agents for flow visualization."""
    from_agent: str
    to_agent: str
    data_type: str  # "analysis", "hypothesis", "impact", "plan"
    data_summary: str
    data_size: int  # Number of items/findings passed
    timestamp: datetime
    communication_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    confidence_score: Optional[float] = None


@dataclass
class OutputChunk:
    """Represents a chunk of agent output for streaming."""
    agent_name: str
    chunk_id: str
    chunk_content: str
    chunk_index: int
    total_chunks: Optional[int] = None
    is_complete: bool = False
    chunk_type: str = "text"  # "text", "analysis", "finding", "conclusion"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentState:
    """State of an individual agent - Enhanced for Phase 2."""
    name: str
    role: AgentRole
    status: AgentStatus = AgentStatus.IDLE
    progress: float = 0.0
    message: Optional[str] = None
    start_time: Optional[datetime] = None
    complete_time: Optional[datetime] = None
    processing_data: Dict[str, Any] = field(default_factory=dict)
    activities: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    confidence_score: Optional[float] = None

    # Phase 2 enhancements
    output_chunks: List[OutputChunk] = field(default_factory=list)
    full_output: Optional[str] = None
    communications_sent: List[AgentCommunication] = field(default_factory=list)
    communications_received: List[AgentCommunication] = field(default_factory=list)
    processing_stage: str = "idle"  # "initializing", "analyzing", "generating", "finalizing"
    estimated_completion: Optional[datetime] = None


@dataclass
class PipelineState:
    """Complete pipeline state - Enhanced for Phase 2."""
    pipeline_id: str
    status: PipelineStatus = PipelineStatus.IDLE
    incident_id: Optional[str] = None
    severity: Optional[Severity] = None
    start_time: Optional[datetime] = None
    current_agent: Optional[str] = None
    overall_progress: float = 0.0
    processed_logs: List[Dict[str, Any]] = field(default_factory=list)
    global_findings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    agents: Dict[str, AgentState] = field(default_factory=dict)

    # Phase 2 enhancements
    communications: List[AgentCommunication] = field(default_factory=list)
    pipeline_config: Dict[str, Any] = field(default_factory=dict)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    estimated_total_time: Optional[int] = None  # seconds
    pause_time: Optional[datetime] = None
    resume_time: Optional[datetime] = None
    pipeline_priority: int = 1  # 1=high, 2=medium, 3=low (for multi-pipeline support)


class PipelineStateManager:
    """Manages pipeline state and WebSocket connections."""

    def __init__(self):
        self.pipelines: Dict[str, PipelineState] = {}
        self.websocket_connections: Set[Any] = set()  # WebSocket connections
        self._lock = asyncio.Lock()

        # Initialize default agents
        self._agent_definitions = [
            ("Initial Analysis", AgentRole.ANALYST),  # Using ANALYST alias (same value as SIGNALS)
            ("Root Cause", AgentRole.RCA),
            ("Impact Assessment", AgentRole.IMPACT),
            ("Solution Planning", AgentRole.MITIGATION)
        ]

    async def create_pipeline(self, pipeline_id: str, incident_id: str, severity: Severity) -> PipelineState:
        """Create a new pipeline instance."""
        async with self._lock:
            pipeline_state = PipelineState(
                pipeline_id=pipeline_id,
                incident_id=incident_id,
                severity=severity,
                start_time=datetime.now(timezone.utc)
            )

            # Initialize agents
            for agent_name, agent_role in self._agent_definitions:
                pipeline_state.agents[agent_name] = AgentState(
                    name=agent_name,
                    role=agent_role
                )

            self.pipelines[pipeline_id] = pipeline_state
            await self._broadcast_update(pipeline_id, "pipeline_created")
            return pipeline_state

    async def start_pipeline(self, pipeline_id: str) -> None:
        """Start pipeline execution."""
        async with self._lock:
            if pipeline_id not in self.pipelines:
                raise ValueError(f"Pipeline {pipeline_id} not found")

            pipeline = self.pipelines[pipeline_id]
            pipeline.status = PipelineStatus.RUNNING
            pipeline.overall_progress = 0.0

            await self._broadcast_update(pipeline_id, "pipeline_started")

    async def update_agent_status(
        self,
        pipeline_id: str,
        agent_name: str,
        status: AgentStatus,
        progress: float = 0.0,
        message: Optional[str] = None,
        processing_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update status of a specific agent."""
        async with self._lock:
            if pipeline_id not in self.pipelines:
                return

            pipeline = self.pipelines[pipeline_id]
            if agent_name not in pipeline.agents:
                return

            agent = pipeline.agents[agent_name]
            old_status = agent.status

            agent.status = status
            agent.progress = progress
            if message:
                agent.message = message
            if processing_data:
                agent.processing_data.update(processing_data)

            # Update timestamps
            if status == AgentStatus.PROCESSING and old_status == AgentStatus.IDLE:
                agent.start_time = datetime.now(timezone.utc)
                pipeline.current_agent = agent_name
            elif status == AgentStatus.COMPLETE:
                agent.complete_time = datetime.now(timezone.utc)
                agent.progress = 1.0

            # Update overall pipeline progress
            self._update_overall_progress(pipeline)

            await self._broadcast_update(pipeline_id, "agent_status_updated")

    async def add_agent_activity(
        self,
        pipeline_id: str,
        agent_name: str,
        activity_type: str,
        description: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an activity for an agent."""
        async with self._lock:
            if pipeline_id not in self.pipelines:
                return

            pipeline = self.pipelines[pipeline_id]
            if agent_name not in pipeline.agents:
                return

            activity = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "activity_type": activity_type,
                "description": description,
                "details": details or {}
            }

            pipeline.agents[agent_name].activities.append(activity)
            await self._broadcast_update(pipeline_id, "agent_activity")

    async def add_agent_finding(
        self,
        pipeline_id: str,
        agent_name: str,
        finding: str,
        confidence: Optional[float] = None
    ) -> None:
        """Add a finding for an agent."""
        async with self._lock:
            if pipeline_id not in self.pipelines:
                return

            pipeline = self.pipelines[pipeline_id]
            if agent_name not in pipeline.agents:
                return

            agent = pipeline.agents[agent_name]
            agent.findings.append(finding)
            if confidence is not None:
                agent.confidence_score = confidence

            # Also add to global findings
            pipeline.global_findings.append(f"{agent_name}: {finding}")

            await self._broadcast_update(pipeline_id, "agent_finding")

    async def add_processed_logs(
        self,
        pipeline_id: str,
        logs: List[Dict[str, Any]]
    ) -> None:
        """Add processed log data to pipeline state."""
        async with self._lock:
            if pipeline_id not in self.pipelines:
                return

            pipeline = self.pipelines[pipeline_id]
            pipeline.processed_logs.extend(logs)

            await self._broadcast_update(pipeline_id, "logs_processed")

    async def complete_pipeline(
        self,
        pipeline_id: str,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """Mark pipeline as completed."""
        async with self._lock:
            if pipeline_id not in self.pipelines:
                return

            pipeline = self.pipelines[pipeline_id]
            pipeline.status = PipelineStatus.COMPLETED if success else PipelineStatus.FAILED
            pipeline.overall_progress = 1.0
            pipeline.current_agent = None

            if error_message:
                pipeline.error_message = error_message

            await self._broadcast_update(pipeline_id, "pipeline_completed")

    async def add_websocket_connection(self, websocket: Any) -> None:
        """Add a WebSocket connection."""
        self.websocket_connections.add(websocket)

    async def remove_websocket_connection(self, websocket: Any) -> None:
        """Remove a WebSocket connection."""
        self.websocket_connections.discard(websocket)

    async def broadcast_update(self, update: dict) -> None:
        """
        Broadcast arbitrary update to all connected WebSocket clients.

        This is a public method for sending custom messages (not tied to pipeline state).
        Used for streaming analysis, custom events, etc.
        """
        if not self.websocket_connections:
            print("⚠️  No WebSocket connections to broadcast to")
            return

        print(f"📡 Broadcasting to {len(self.websocket_connections)} clients: {update.get('type', 'unknown')}")

        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_json(update)
                print(f"   ✅ Sent to WebSocket client")
            except Exception as e:
                print(f"   ❌ Failed to send to WebSocket: {e}")
                disconnected.append(websocket)

        # Remove disconnected clients
        for ws in disconnected:
            self.websocket_connections.discard(ws)
            print(f"   🗑️  Removed disconnected WebSocket")

    def get_pipeline_state(self, pipeline_id: str) -> Optional[PipelineState]:
        """Get current pipeline state."""
        return self.pipelines.get(pipeline_id)

    def _update_overall_progress(self, pipeline: PipelineState) -> None:
        """Update overall pipeline progress based on agent progress."""
        if not pipeline.agents:
            return

        total_progress = sum(agent.progress for agent in pipeline.agents.values())
        pipeline.overall_progress = total_progress / len(pipeline.agents)

    async def _broadcast_update(self, pipeline_id: str, event_type: str) -> None:
        """Broadcast pipeline update to all connected WebSockets."""
        if not self.websocket_connections:
            return

        pipeline = self.pipelines.get(pipeline_id)
        if not pipeline:
            return

        # Convert pipeline state to WebSocket message format - Enhanced for Phase 2
        message = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_id": pipeline_id,
            "status": pipeline.status.value,
            "overall_progress": pipeline.overall_progress,
            "current_step": self._get_current_step_description(pipeline),
            "agents": [
                {
                    "name": agent.name,
                    "status": agent.status.value,
                    "progress": agent.progress,
                    "message": agent.message,
                    "current_activity": agent.activities[-1]["description"] if agent.activities else None,
                    "activities": agent.activities[-5:],  # Last 5 activities
                    "findings": agent.findings,
                    "confidence_score": agent.confidence_score,
                    "processing_data": agent.processing_data,

                    # Phase 2 enhancements
                    "processing_stage": agent.processing_stage,
                    "full_output": agent.full_output,
                    "output_chunks": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "chunk_content": chunk.chunk_content,
                            "chunk_type": chunk.chunk_type,
                            "chunk_index": chunk.chunk_index,
                            "is_complete": chunk.is_complete,
                            "timestamp": chunk.timestamp.isoformat()
                        }
                        for chunk in agent.output_chunks[-3:]  # Last 3 chunks
                    ],
                    "communications_sent": len(agent.communications_sent),
                    "communications_received": len(agent.communications_received),
                    "estimated_completion": agent.estimated_completion.isoformat() if agent.estimated_completion else None
                }
                for agent in pipeline.agents.values()
            ],
            "recent_activities": self._get_recent_activities(pipeline),
            "global_findings": pipeline.global_findings,
            "processed_logs": pipeline.processed_logs[-10:],  # Last 10 log entries
            "error_message": pipeline.error_message,

            # Phase 2 enhancements
            "communications": [
                {
                    "communication_id": comm.communication_id,
                    "from_agent": comm.from_agent,
                    "to_agent": comm.to_agent,
                    "data_type": comm.data_type,
                    "data_summary": comm.data_summary,
                    "data_size": comm.data_size,
                    "confidence_score": comm.confidence_score,
                    "timestamp": comm.timestamp.isoformat()
                }
                for comm in pipeline.communications[-10:]  # Last 10 communications
            ],
            "pipeline_config": pipeline.pipeline_config,
            "execution_metadata": pipeline.execution_metadata,
            "estimated_total_time": pipeline.estimated_total_time,
            "pipeline_priority": pipeline.pipeline_priority,
            "pause_time": pipeline.pause_time.isoformat() if pipeline.pause_time else None,
            "resume_time": pipeline.resume_time.isoformat() if pipeline.resume_time else None,

            # Multi-pipeline context
            "active_pipelines": len(self.get_all_active_pipelines()),
        }

        # Send to all connections
        disconnected = set()
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception:
                # Mark for removal
                disconnected.add(websocket)

        # Remove disconnected sockets
        for websocket in disconnected:
            self.websocket_connections.discard(websocket)

    def _get_current_step_description(self, pipeline: PipelineState) -> str:
        """Get human-readable description of current pipeline step."""
        if pipeline.status == PipelineStatus.IDLE:
            return "Pipeline ready to start"
        elif pipeline.status == PipelineStatus.STARTING:
            return "Initializing agents..."
        elif pipeline.status == PipelineStatus.COMPLETED:
            return "Pipeline completed successfully"
        elif pipeline.status == PipelineStatus.FAILED:
            return f"Pipeline failed: {pipeline.error_message or 'Unknown error'}"
        elif pipeline.current_agent:
            return f"{pipeline.current_agent}: Processing incident data..."
        else:
            return "Pipeline running..."

    def _get_recent_activities(self, pipeline: PipelineState, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent activities across all agents."""
        all_activities = []

        for agent in pipeline.agents.values():
            for activity in agent.activities:
                activity_with_agent = dict(activity)
                activity_with_agent["agent_name"] = agent.name
                all_activities.append(activity_with_agent)

        # Sort by timestamp and return most recent
        all_activities.sort(key=lambda x: x["timestamp"], reverse=True)
        return all_activities[:limit]

    # ==================== Phase 2: Enhanced Agent Communication ====================

    async def add_agent_communication(
        self,
        pipeline_id: str,
        from_agent: str,
        to_agent: str,
        data_type: str,
        data_summary: str,
        data_size: int = 0,
        confidence_score: Optional[float] = None
    ) -> str:
        """Track communication between agents for flow visualization."""
        async with self._lock:
            if pipeline_id not in self.pipelines:
                return ""

            pipeline = self.pipelines[pipeline_id]

            communication = AgentCommunication(
                from_agent=from_agent,
                to_agent=to_agent,
                data_type=data_type,
                data_summary=data_summary,
                data_size=data_size,
                timestamp=datetime.now(timezone.utc),
                confidence_score=confidence_score
            )

            # Add to pipeline communications
            pipeline.communications.append(communication)

            # Track on agent states
            if from_agent in pipeline.agents:
                pipeline.agents[from_agent].communications_sent.append(communication)
            if to_agent in pipeline.agents:
                pipeline.agents[to_agent].communications_received.append(communication)

            await self._broadcast_update(pipeline_id, "agent_communication")
            return communication.communication_id

    async def stream_agent_output_chunk(
        self,
        pipeline_id: str,
        agent_name: str,
        chunk_content: str,
        chunk_type: str = "text",
        is_complete: bool = False,
        total_chunks: Optional[int] = None
    ) -> None:
        """Stream agent output in chunks for real-time display."""
        async with self._lock:
            if pipeline_id not in self.pipelines:
                return

            pipeline = self.pipelines[pipeline_id]
            if agent_name not in pipeline.agents:
                return

            agent = pipeline.agents[agent_name]

            chunk = OutputChunk(
                agent_name=agent_name,
                chunk_id=str(uuid.uuid4()),
                chunk_content=chunk_content,
                chunk_index=len(agent.output_chunks),
                total_chunks=total_chunks,
                is_complete=is_complete,
                chunk_type=chunk_type
            )

            agent.output_chunks.append(chunk)

            # Update full output
            if agent.full_output is None:
                agent.full_output = chunk_content
            else:
                agent.full_output += chunk_content

            await self._broadcast_update(pipeline_id, "agent_output_chunk")

    async def update_agent_processing_stage(
        self,
        pipeline_id: str,
        agent_name: str,
        stage: str,
        estimated_completion: Optional[datetime] = None
    ) -> None:
        """Update agent's current processing stage for detailed tracking."""
        async with self._lock:
            if pipeline_id not in self.pipelines:
                return

            pipeline = self.pipelines[pipeline_id]
            if agent_name not in pipeline.agents:
                return

            agent = pipeline.agents[agent_name]
            agent.processing_stage = stage
            agent.estimated_completion = estimated_completion

            await self._broadcast_update(pipeline_id, "agent_stage_updated")

    # ==================== Phase 2: Pipeline Control ====================

    async def pause_pipeline(self, pipeline_id: str) -> bool:
        """Pause a running pipeline."""
        async with self._lock:
            if pipeline_id not in self.pipelines:
                return False

            pipeline = self.pipelines[pipeline_id]
            if pipeline.status == PipelineStatus.RUNNING:
                pipeline.status = PipelineStatus.IDLE  # Using IDLE for pause state
                pipeline.pause_time = datetime.now(timezone.utc)

                # Pause all agents
                for agent in pipeline.agents.values():
                    if agent.status == AgentStatus.PROCESSING:
                        agent.status = AgentStatus.PAUSED

                await self._broadcast_update(pipeline_id, "pipeline_paused")
                return True
            return False

    async def resume_pipeline(self, pipeline_id: str) -> bool:
        """Resume a paused pipeline."""
        async with self._lock:
            if pipeline_id not in self.pipelines:
                return False

            pipeline = self.pipelines[pipeline_id]
            if pipeline.pause_time is not None:
                pipeline.status = PipelineStatus.RUNNING
                pipeline.resume_time = datetime.now(timezone.utc)

                # Resume paused agents
                for agent in pipeline.agents.values():
                    if agent.status == AgentStatus.PAUSED:
                        agent.status = AgentStatus.PROCESSING

                await self._broadcast_update(pipeline_id, "pipeline_resumed")
                return True
            return False

    async def update_pipeline_config(
        self,
        pipeline_id: str,
        config_updates: Dict[str, Any]
    ) -> bool:
        """Update pipeline configuration during execution."""
        async with self._lock:
            if pipeline_id not in self.pipelines:
                return False

            pipeline = self.pipelines[pipeline_id]
            pipeline.pipeline_config.update(config_updates)

            await self._broadcast_update(pipeline_id, "pipeline_config_updated")
            return True

    # ==================== Phase 2: Multi-Pipeline Support ====================

    def get_all_active_pipelines(self) -> List[str]:
        """Get list of all active pipeline IDs."""
        return [
            pipeline_id for pipeline_id, pipeline in self.pipelines.items()
            if pipeline.status in [PipelineStatus.RUNNING, PipelineStatus.STARTING]
        ]

    def get_pipeline_summary(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of pipeline state for multi-pipeline dashboard."""
        pipeline = self.pipelines.get(pipeline_id)
        if not pipeline:
            return None

        return {
            "pipeline_id": pipeline_id,
            "status": pipeline.status.value,
            "incident_id": pipeline.incident_id,
            "severity": pipeline.severity.value if pipeline.severity else None,
            "overall_progress": pipeline.overall_progress,
            "start_time": pipeline.start_time.isoformat() if pipeline.start_time else None,
            "current_agent": pipeline.current_agent,
            "priority": pipeline.pipeline_priority,
            "agent_count": len(pipeline.agents),
            "communications_count": len(pipeline.communications),
            "error_message": pipeline.error_message
        }

    async def set_pipeline_priority(self, pipeline_id: str, priority: int) -> bool:
        """Set pipeline priority for multi-pipeline management."""
        async with self._lock:
            if pipeline_id not in self.pipelines:
                return False

            self.pipelines[pipeline_id].pipeline_priority = priority
            await self._broadcast_update(pipeline_id, "pipeline_priority_updated")
            return True


# Global instance
_pipeline_state_manager: Optional[PipelineStateManager] = None


def get_pipeline_state_manager() -> PipelineStateManager:
    """Get global pipeline state manager instance."""
    global _pipeline_state_manager
    if _pipeline_state_manager is None:
        _pipeline_state_manager = PipelineStateManager()
    return _pipeline_state_manager
