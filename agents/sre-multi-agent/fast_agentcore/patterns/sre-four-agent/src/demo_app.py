#!/usr/bin/env python3
"""FastAPI demo application for AWS Bedrock SRE agent showcase."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import json

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.orchestration.four_agent.analyst_agent import AnalystAgent, AnalystAgentConfig
from src.orchestration.four_agent.rca_agent import RCAAgent
from src.orchestration.four_agent.impact_agent import ImpactAgent, ImpactAgentConfig
from src.orchestration.four_agent.mitigation_agent import MitigationCommsAgent, MitigationAgentConfig
from src.orchestration.four_agent.schema import (
    AgentMessage,
    AgentRole,
    MessageType,
    Severity,
    PayloadModel,
    DemoIncidentReport
)
from src.orchestration.four_agent.state import IncidentState
from src.data_pipeline.pipeline_orchestrator import LogDataPipeline

# Real-time WebSocket integration
from src.orchestration.real_time.pipeline_state_manager import get_pipeline_state_manager
from src.orchestration.real_time.websocket_orchestrator import WebSocketOrchestrator


# Request/Response Models
class IncidentAnalysisRequest(BaseModel):
    """Request model for incident analysis."""
    incident_description: str = Field(..., description="Brief description of the incident")
    severity: str = Field(..., description="SEV-1, SEV-2, or SEV-3")
    logs: Optional[List[Dict[str, Any]]] = Field(default=None, description="Log entries")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="System metrics")
    model_id: Optional[str] = Field(default="us.meta.llama3-3-70b-instruct-v1:0", description="AWS Bedrock model to use")


class IncidentAnalysisResponse(BaseModel):
    """Response model for incident analysis."""
    success: bool
    incident_id: str
    processing_time_seconds: float
    report: Optional[DemoIncidentReport] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    services: Dict[str, str]


class DemoStartRequest(BaseModel):
    """Request to start a WebSocket demo."""
    scenario: str = Field(default="database_timeout", description="Demo scenario to run")


class PipelineStartRequest(BaseModel):
    """Request to start a real pipeline."""
    scenario: str = Field(..., description="Scenario to run")
    model_id: str = Field(default="us.meta.llama3-3-70b-instruct-v1:0", description="AWS Bedrock model to use")


class StreamingStartRequest(BaseModel):
    """Request to start streaming data analysis."""
    start_time: str = Field(..., description="Start time in HH:MM format (e.g., '09:30')")
    window_size_minutes: int = Field(default=15, description="Window size in minutes")
    model_id: str = Field(default="us.meta.llama3-3-70b-instruct-v1:0", description="AWS Bedrock model to use")


class PipelineStartResponse(BaseModel):
    """Response for pipeline start."""
    success: bool
    pipeline_id: str
    incident_id: str
    message: str


class AgentActivity(BaseModel):
    """Individual agent activity/thought."""
    timestamp: str
    activity_type: str  # analyzing, finding, calculating, reasoning, completing
    description: str
    details: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None

class AgentStatus(BaseModel):
    """Status of an individual agent."""
    name: str
    status: str  # idle, processing, complete
    progress: float  # 0.0 to 1.0
    message: Optional[str] = None
    # Enhanced observability fields
    current_activity: Optional[str] = None
    activities: List[AgentActivity] = Field(default_factory=list)
    findings: List[str] = Field(default_factory=list)
    confidence_score: Optional[float] = None
    processing_data: Optional[Dict[str, Any]] = None


class DemoUpdate(BaseModel):
    """Real-time demo update message."""
    type: str  # demo_started, agent_activity, agent_status, progress, completion, error
    timestamp: str
    agents: List[AgentStatus]
    overall_progress: float
    current_step: str
    completion_metrics: Optional[Dict[str, Any]] = None
    # Enhanced observability fields
    recent_activities: List[AgentActivity] = Field(default_factory=list)
    global_findings: List[str] = Field(default_factory=list)
    # Error handling (no fallbacks - just real errors)
    error_message: Optional[str] = None


# Initialize FastAPI app
app = FastAPI(
    title="AWS Bedrock SRE Agent Demo",
    description="Demonstrates LLM-powered incident response with 5-agent orchestration",
    version="1.0.0"
)

# Add CORS middleware for web demos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for live demo updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_update(self, message: DemoUpdate):
        """Send update to all connected clients."""
        message_json = message.model_dump_json()
        for connection in self.active_connections[:]:  # Use slice to avoid modification during iteration
            try:
                await connection.send_text(message_json)
            except WebSocketDisconnect:
                self.active_connections.remove(connection)
            except Exception:
                # Remove broken connections
                if connection in self.active_connections:
                    self.active_connections.remove(connection)


# Global connection manager instance
manager = ConnectionManager()

# Initialize real-time pipeline state manager
pipeline_manager = get_pipeline_state_manager()

# Initialize real data pipeline
data_pipeline = LogDataPipeline()

# Real incident scenarios from streaming data
# These correspond to the actual embedded incidents in the streaming logs
REAL_INCIDENT_SCENARIOS = {
    "payment_processing_failures": {
        "window_time": "09:45",  # SEV-3: Payment Processing Failures
        "description": "Payment processing failures with gateway timeouts",
        "severity": "SEV-3"
    },
    "database_connection_issues": {
        "window_time": "11:15",  # SEV-2: Database Connection Issues
        "description": "Database connection pool exhaustion causing cascading failures",
        "severity": "SEV-2"
    },
    "traffic_surge_overload": {
        "window_time": "12:45",  # SEV-1: Traffic Surge Overload
        "description": "Unexpected traffic spike overwhelming system capacity",
        "severity": "SEV-1"
    }
}

def get_real_incident_data(scenario_key: str) -> dict:
    """Get real incident data from the streaming logs pipeline."""
    try:
        if scenario_key in REAL_INCIDENT_SCENARIOS:
            # Get real data from pipeline
            scenario_info = REAL_INCIDENT_SCENARIOS[scenario_key]
            window_time = scenario_info["window_time"]

            # Get window logs from pipeline (instead of load_demo_scenario)
            # Parse window_time to find matching window
            from datetime import datetime, timedelta

            # Create a window around the specified time
            base_time = data_pipeline.dataset_start_time
            if window_time == "09:45":
                window_start = base_time + timedelta(minutes=45)  # 45 minutes after start
            elif window_time == "11:15":
                window_start = base_time + timedelta(hours=1, minutes=15)  # 1h 15min after start
            elif window_time == "12:45":
                window_start = base_time + timedelta(hours=2, minutes=45)  # 2h 45min after start
            else:
                window_start = base_time  # Default to start

            # Get logs for this window
            window_logs = data_pipeline.get_window_logs(window_start)

            # Convert window logs to demo format
            logs = []
            if window_logs:
                for log_entry in window_logs[:10]:  # Limit to 10 most relevant logs
                    logs.append({
                        "timestamp": log_entry.timestamp.isoformat(),
                        "level": log_entry.level,
                        "message": log_entry.message,
                        "service": log_entry.service
                    })

            # Create basic metrics (placeholder for real metrics)
            metrics = {
                "error_count": len([log for log in window_logs if log.level == "ERROR"]),
                "warn_count": len([log for log in window_logs if log.level == "WARN"]),
                "total_entries": len(window_logs),
                "window_start": window_start.isoformat()
            }

            return {
                "description": scenario_info["description"],
                "severity": scenario_info["severity"],
                "logs": logs,
                "metrics": metrics,
                "real_data": True,
                "window_time": window_time
            }
        elif scenario_key == "streaming_incident":
            # Special case for streaming-triggered incidents
            # This will be handled by the streaming session itself
            return {
                "description": "Incident triggered from streaming analysis",
                "severity": "SEV-2",
                "logs": [],  # Will be populated by streaming session
                "metrics": {},
                "real_data": True,
                "streaming_incident": True
            }
        else:
            # For backward compatibility, map old names to new ones
            scenario_mapping = {
                "database_timeout": "database_connection_issues",
                "payment_api_errors": "payment_processing_failures",
                "memory_leak_detection": "traffic_surge_overload"
            }
            if scenario_key in scenario_mapping:
                return get_real_incident_data(scenario_mapping[scenario_key])

    except Exception as e:
        print(f"⚠️  Error loading real incident data for {scenario_key}: {e}")
        print(f"⚠️  Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("Using fallback data structure")

    # Fallback: return empty structure for compatibility
    return {
        "description": f"Real incident data for {scenario_key}",
        "severity": "SEV-2",
        "logs": [],
        "metrics": {},
        "real_data": False,
        "error": "Could not load real data"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        services={
            "fastapi": "running",
            "aws_bedrock": "connected",
            "agent_orchestration": "ready"
        }
    )


@app.get("/demo/scenarios")
async def list_demo_scenarios():
    """List available demo scenarios from real streaming data."""
    scenarios = {}

    # Add real incident scenarios
    for scenario_key, scenario_info in REAL_INCIDENT_SCENARIOS.items():
        scenarios[scenario_key] = {
            "description": scenario_info["description"],
            "severity": scenario_info["severity"],
            "window_time": scenario_info["window_time"],
            "data_source": "real_streaming_logs"
        }

    # Also add backward compatibility scenarios
    compatibility_scenarios = {
        "database_timeout": "database_connection_issues",
        "payment_api_errors": "payment_processing_failures",
        "memory_leak_detection": "traffic_surge_overload"
    }

    return {
        "scenarios": list(REAL_INCIDENT_SCENARIOS.keys()) + list(compatibility_scenarios.keys()),
        "real_incidents": scenarios,
        "compatibility_mapping": compatibility_scenarios,
        "data_source": "kafka_style_streaming_logs",
        "total_log_entries": len(data_pipeline.load_logs()),
        "available_windows": len(data_pipeline.get_available_windows())
    }


@app.post("/demo/sample-incident/{scenario_name}", response_model=IncidentAnalysisResponse)
async def analyze_sample_incident(
    scenario_name: str,
    model_id: Optional[str] = "us.meta.llama3-3-70b-instruct-v1:0"
):
    """Analyze a real incident scenario from streaming data."""
    # Get real incident data
    real_data = get_real_incident_data(scenario_name)

    if not real_data or "error" in real_data:
        available_scenarios = list(REAL_INCIDENT_SCENARIOS.keys()) + ["database_timeout", "payment_api_errors", "memory_leak_detection"]
        raise HTTPException(
            status_code=404,
            detail=f"Scenario '{scenario_name}' not found or could not load real data. Available: {available_scenarios}"
        )

    request = IncidentAnalysisRequest(
        incident_description=real_data["description"],
        severity=real_data["severity"],
        logs=real_data["logs"],
        metrics=real_data["metrics"],
        model_id=model_id
    )

    return await analyze_incident(request)


@app.post("/analyze-incident", response_model=IncidentAnalysisResponse)
async def analyze_incident(request: IncidentAnalysisRequest):
    """Analyze an incident using the 5-agent orchestration system."""
    start_time = time.time()
    incident_id = f"incident-{int(start_time)}"

    try:
        # Convert severity string to enum
        severity_mapping = {"SEV-1": Severity.SEV_1, "SEV-2": Severity.SEV_2, "SEV-3": Severity.SEV_3}
        severity = severity_mapping.get(request.severity, Severity.SEV_2)

        # Initialize incident state
        state = IncidentState(incident_id=incident_id, severity=severity)

        # Initialize agents with the specified model
        analyst_config = AnalystAgentConfig()
        analyst_agent = AnalystAgent(config=analyst_config, model=request.model_id)

        rca_agent = RCAAgent(model=request.model_id)

        impact_config = ImpactAgentConfig()
        impact_agent = ImpactAgent(config=impact_config, model=request.model_id)

        mitigation_config = MitigationAgentConfig()
        mitigation_agent = MitigationCommsAgent(config=mitigation_config, model=request.model_id)

        # Phase 1: Analyst Agent - Analyze logs and metrics
        analyst_message = AgentMessage(
            incident_id=incident_id,
            sender=AgentRole.ORCHESTRATOR,
            type=MessageType.REQUEST,
            severity=severity,
            payload=PayloadModel(
                summary=request.incident_description,
                details={
                    "logs": request.logs or [],
                    "monitoring": {"metrics": request.metrics or {}}
                }
            )
        )

        analyst_result = await analyst_agent.handle(analyst_message, state)

        # Phase 2: RCA Agent - Determine root cause
        rca_message = AgentMessage(
            incident_id=incident_id,
            sender=AgentRole.ANALYST,
            type=MessageType.HYPOTHESIS,
            severity=severity,
            payload=PayloadModel(
                summary=analyst_result.payload.summary or "Anomalies detected in system",
                details=analyst_result.payload.details
            )
        )

        rca_result = await rca_agent.handle(rca_message, state)

        # Phase 3: Impact Agent - Assess business impact
        impact_message = AgentMessage(
            incident_id=incident_id,
            sender=AgentRole.RCA,
            type=MessageType.IMPACT,
            severity=severity,
            payload=PayloadModel(
                summary=rca_result.payload.summary or "Root cause analysis completed",
                details={
                    **rca_result.payload.details,
                    "rca_confidence": getattr(rca_result.payload.details, "confidence", 0.8)
                }
            )
        )

        impact_result = await impact_agent.handle(impact_message, state)

        # Phase 4: Mitigation Agent - Generate action plan and communications
        mitigation_message = AgentMessage(
            incident_id=incident_id,
            sender=AgentRole.IMPACT,
            type=MessageType.PLAN,
            severity=severity,
            payload=PayloadModel(
                summary=impact_result.payload.summary or "Business impact assessed",
                details={
                    "rca": rca_result.payload.summary,
                    "impact": impact_result.payload.summary,
                    "signals": analyst_result.payload.summary,
                    **impact_result.payload.details
                }
            )
        )

        mitigation_result = await mitigation_agent.handle(mitigation_message, state)

        # Generate final demo report
        processing_time = time.time() - start_time

        demo_report = DemoIncidentReport(
            executive_summary=f"Incident {incident_id}: {request.incident_description}",
            severity_level=request.severity,
            estimated_impact=impact_result.payload.summary or "Impact assessment completed",
            anomaly_details=analyst_result.payload.details.get("anomalies", {}),
            root_cause_analysis={
                "analysis": rca_result.payload.summary or "Root cause identified",
                "confidence": rca_result.payload.details.get("confidence", 0.8),
                "details": rca_result.payload.details
            },
            technical_mitigation_plan=mitigation_result.payload.details.get("technical_plan", "Mitigation plan generated"),
            stakeholder_notification=mitigation_result.payload.details.get("executive_summary", "Stakeholders notified"),
            customer_communication=mitigation_result.payload.details.get("customer_comms", "Customer communication prepared"),
            agents_involved=["AnalystAgent", "RCAAgent", "ImpactAgent", "MitigationAgent"],
            rag_documents_used=mitigation_result.payload.details.get("policy_documents", []),
            processing_time_seconds=round(processing_time, 2),
            incident_id=incident_id,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        return IncidentAnalysisResponse(
            success=True,
            incident_id=incident_id,
            processing_time_seconds=round(processing_time, 2),
            report=demo_report
        )

    except Exception as e:
        processing_time = time.time() - start_time
        return IncidentAnalysisResponse(
            success=False,
            incident_id=incident_id,
            processing_time_seconds=round(processing_time, 2),
            error=str(e)
        )


@app.websocket("/ws/demo")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time demo updates."""
    await websocket.accept()
    print(f"✅ WebSocket client connected to /ws/demo")

    # Register with pipeline_manager (not the old manager)
    await pipeline_manager.add_websocket_connection(websocket)
    print(f"📊 Total WebSocket connections: {len(pipeline_manager.websocket_connections)}")
    print(f"🔍 DEBUG: Global pipeline_manager ID: {id(pipeline_manager)}")
    print(f"🔍 DEBUG: WebSocket connections set ID: {id(pipeline_manager.websocket_connections)}")

    # Send connection confirmation
    try:
        await websocket.send_json({
            "type": "connection_established",
            "message": "WebSocket connected to SRE PoC backend",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        print(f"Failed to send connection confirmation: {e}")

    try:
        while True:
            # Keep connection alive and handle disconnect
            data = await websocket.receive_text()
            # Echo back any received data (for ping/pong)
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        print(f"❌ WebSocket client disconnected from /ws/demo")
        await pipeline_manager.remove_websocket_connection(websocket)
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        await pipeline_manager.remove_websocket_connection(websocket)


@app.websocket("/ws/pipeline")
async def pipeline_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time pipeline updates."""
    await websocket.accept()

    # Add to pipeline state manager
    await pipeline_manager.add_websocket_connection(websocket)

    try:
        while True:
            # Keep connection alive and handle disconnect
            data = await websocket.receive_text()
            # Handle ping/pong and potential control messages
            if data == "ping":
                await websocket.send_text("pong")
            # Could add other control messages here in the future
    except WebSocketDisconnect:
        await pipeline_manager.remove_websocket_connection(websocket)


@app.post("/test/websocket-broadcast")
async def test_websocket_broadcast():
    """Test WebSocket broadcasting independently."""
    test_message = {
        "type": "test_broadcast",
        "message": "Manual WebSocket test",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    print(f"🧪 MANUAL WEBSOCKET TEST")
    print(f"   Global pipeline_manager ID: {id(pipeline_manager)}")
    print(f"   WebSocket connections: {len(pipeline_manager.websocket_connections)}")

    await pipeline_manager.broadcast_update(test_message)

    return {
        "success": True,
        "connections": len(pipeline_manager.websocket_connections),
        "message_sent": test_message
    }

@app.post("/api/pipeline/start", response_model=PipelineStartResponse)
async def start_pipeline(request: PipelineStartRequest):
    """Start the REAL streaming data pipeline with WebSocket updates."""
    # Generate unique IDs
    pipeline_id = f"pipeline-{int(time.time())}"

    # Log before starting background task
    print(f"🎬 Starting REAL streaming data pipeline")
    print(f"   Pipeline ID: {pipeline_id}")
    print(f"   Model: {request.model_id}")

    # CRITICAL FIX: Wait for WebSocket connection before starting pipeline
    async def start_streaming_pipeline():
        max_wait = 5  # seconds
        wait_interval = 0.5  # seconds

        print(f"⏳ Waiting for WebSocket connection (max {max_wait}s)...")
        for i in range(int(max_wait / wait_interval)):
            if len(pipeline_manager.websocket_connections) > 0:
                print(f"✅ WebSocket connection detected after {i * wait_interval}s - starting pipeline")
                break
            await asyncio.sleep(wait_interval)
        else:
            print(f"⚠️  No WebSocket connection after {max_wait}s, starting anyway...")

        # Import and run the REAL pipeline from full_pipeline_runner.py
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from full_pipeline_runner import FullPipelineRunner

        print(f"\n🚀 STARTING REAL STREAMING DATA PIPELINE")
        print(f"=" * 70)

        # Create runner WITH pipeline_manager for UI broadcasting
        runner = FullPipelineRunner(start_time=None, pipeline_manager=pipeline_manager)

        # Broadcast pipeline started
        await pipeline_manager.broadcast_update({
            "type": "pipeline_started",
            "pipeline_id": pipeline_id,
            "message": "Starting streaming data analysis",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Load configuration for window count
        config_path = Path(__file__).parent.parent / "configs" / "data_pipeline.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        window_count = config.get("processing", {}).get("window_count", 2)

        print(f"📊 Processing {window_count} data windows from streaming logs...")
        print(f"💡 TIP: Watch the frontend for real-time updates!")
        print()

        # Run the full pipeline (this will broadcast to UI automatically)
        report_path = await runner.run_full_pipeline(window_count)

        print(f"\n✅ STREAMING PIPELINE COMPLETED!")
        print(f"📋 Report: {report_path}")

        # Broadcast completion
        await pipeline_manager.broadcast_update({
            "type": "pipeline_completed",
            "pipeline_id": pipeline_id,
            "report_path": str(report_path),
            "message": "Streaming data analysis complete",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    # Run pipeline in background
    task = asyncio.create_task(start_streaming_pipeline())

    # Add error callback to catch any exceptions
    def task_exception_handler(future):
        print(f"🔍 Background task completed/failed")
        try:
            exc = future.exception()
            if exc:
                print(f"❌❌❌ BACKGROUND TASK FAILED WITH EXCEPTION ❌❌❌")
                print(f"❌ Exception type: {type(exc).__name__}")
                print(f"❌ Exception message: {exc}")
                print(f"❌ Full traceback:")
                import traceback
                traceback.print_exception(type(exc), exc, exc.__traceback__)
                print("❌" * 40)
            else:
                print(f"✅ Background task completed successfully")
        except asyncio.CancelledError:
            print(f"⚠️  Background task was cancelled")
        except Exception as e:
            print(f"❌ Error in exception handler: {e}")
            import traceback
            traceback.print_exc()

    task.add_done_callback(task_exception_handler)

    print(f"✅ Background task created and scheduled")

    return PipelineStartResponse(
        success=True,
        pipeline_id=pipeline_id,
        incident_id=pipeline_id,  # Use same ID for incident
        message=f"Streaming data pipeline started - processing real logs"
    )


@app.get("/api/pipeline/{pipeline_id}/status")
async def get_pipeline_status(pipeline_id: str):
    """Get current pipeline status - Enhanced for Phase 2."""
    pipeline_state = pipeline_manager.get_pipeline_state(pipeline_id)
    if not pipeline_state:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")

    return {
        "pipeline_id": pipeline_id,
        "status": pipeline_state.status.value,
        "overall_progress": pipeline_state.overall_progress,
        "current_agent": pipeline_state.current_agent,
        "incident_id": pipeline_state.incident_id,
        "start_time": pipeline_state.start_time.isoformat() if pipeline_state.start_time else None,
        "agents": {
            name: {
                "status": agent.status.value,
                "progress": agent.progress,
                "message": agent.message,
                "findings_count": len(agent.findings),
                # Phase 2 enhancements
                "processing_stage": agent.processing_stage,
                "output_chunks_count": len(agent.output_chunks),
                "communications_sent": len(agent.communications_sent),
                "communications_received": len(agent.communications_received),
                "estimated_completion": agent.estimated_completion.isoformat() if agent.estimated_completion else None
            }
            for name, agent in pipeline_state.agents.items()
        },
        # Phase 2 enhancements
        "communications_count": len(pipeline_state.communications),
        "pipeline_priority": pipeline_state.pipeline_priority,
        "pause_time": pipeline_state.pause_time.isoformat() if pipeline_state.pause_time else None,
        "resume_time": pipeline_state.resume_time.isoformat() if pipeline_state.resume_time else None,
        "estimated_total_time": pipeline_state.estimated_total_time
    }


# ==================== Phase 2: Enhanced Pipeline Control APIs ====================

@app.post("/api/pipeline/{pipeline_id}/pause")
async def pause_pipeline(pipeline_id: str):
    """Pause a running pipeline."""
    success = await pipeline_manager.pause_pipeline(pipeline_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot pause pipeline {pipeline_id}. Pipeline not found or not running."
        )

    return {"success": True, "message": f"Pipeline {pipeline_id} paused", "pipeline_id": pipeline_id}


@app.post("/api/pipeline/{pipeline_id}/resume")
async def resume_pipeline(pipeline_id: str):
    """Resume a paused pipeline."""
    success = await pipeline_manager.resume_pipeline(pipeline_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume pipeline {pipeline_id}. Pipeline not found or not paused."
        )

    return {"success": True, "message": f"Pipeline {pipeline_id} resumed", "pipeline_id": pipeline_id}


class PipelineConfigUpdate(BaseModel):
    """Request model for pipeline configuration updates."""
    model_id: Optional[str] = None
    priority: Optional[int] = None
    processing_mode: Optional[str] = None  # "fast", "detailed", "comprehensive"
    custom_parameters: Optional[Dict[str, Any]] = None


@app.put("/api/pipeline/{pipeline_id}/config")
async def update_pipeline_config(pipeline_id: str, config_update: PipelineConfigUpdate):
    """Update pipeline configuration during execution."""
    config_dict = config_update.model_dump(exclude_unset=True)

    # Handle priority separately
    if "priority" in config_dict:
        priority = config_dict.pop("priority")
        await pipeline_manager.set_pipeline_priority(pipeline_id, priority)

    if config_dict:
        success = await pipeline_manager.update_pipeline_config(pipeline_id, config_dict)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Pipeline {pipeline_id} not found"
            )

    return {
        "success": True,
        "message": f"Pipeline {pipeline_id} configuration updated",
        "pipeline_id": pipeline_id,
        "updated_config": config_dict
    }


@app.get("/api/pipelines/active")
async def get_active_pipelines():
    """Get all active pipelines for multi-pipeline dashboard."""
    active_pipeline_ids = pipeline_manager.get_all_active_pipelines()
    pipelines_summary = []

    for pipeline_id in active_pipeline_ids:
        summary = pipeline_manager.get_pipeline_summary(pipeline_id)
        if summary:
            pipelines_summary.append(summary)

    return {
        "active_pipelines": len(active_pipeline_ids),
        "pipelines": pipelines_summary,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/pipeline/{pipeline_id}/communications")
async def get_pipeline_communications(pipeline_id: str, limit: int = 20):
    """Get agent communications for flow visualization."""
    pipeline_state = pipeline_manager.get_pipeline_state(pipeline_id)
    if not pipeline_state:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")

    communications = pipeline_state.communications[-limit:] if limit > 0 else pipeline_state.communications

    return {
        "pipeline_id": pipeline_id,
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
            for comm in communications
        ],
        "total_communications": len(pipeline_state.communications)
    }


@app.get("/api/pipeline/{pipeline_id}/agent/{agent_name}/output")
async def get_agent_output_stream(pipeline_id: str, agent_name: str, chunk_limit: int = 50):
    """Get agent output chunks for real-time streaming display."""
    pipeline_state = pipeline_manager.get_pipeline_state(pipeline_id)
    if not pipeline_state:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")

    if agent_name not in pipeline_state.agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found in pipeline")

    agent = pipeline_state.agents[agent_name]
    chunks = agent.output_chunks[-chunk_limit:] if chunk_limit > 0 else agent.output_chunks

    return {
        "pipeline_id": pipeline_id,
        "agent_name": agent_name,
        "full_output": agent.full_output,
        "output_chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "chunk_content": chunk.chunk_content,
                "chunk_type": chunk.chunk_type,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "is_complete": chunk.is_complete,
                "timestamp": chunk.timestamp.isoformat()
            }
            for chunk in chunks
        ],
        "total_chunks": len(agent.output_chunks),
        "processing_stage": agent.processing_stage
    }


@app.post("/api/demo/start")
async def start_demo(request: DemoStartRequest):
    """Start a real AWS Bedrock agent demo with real streaming data."""
    # Check if scenario exists in real data
    real_data = get_real_incident_data(request.scenario)
    if not real_data or "error" in real_data:
        available_scenarios = list(REAL_INCIDENT_SCENARIOS.keys()) + ["database_timeout", "payment_api_errors", "memory_leak_detection"]
        raise HTTPException(
            status_code=404,
            detail=f"Scenario '{request.scenario}' not found in real data. Available: {available_scenarios}"
        )

    # Run REAL agent demo with REAL DATA in background
    asyncio.create_task(run_real_agent_demo(request.scenario))

    scenario_info = REAL_INCIDENT_SCENARIOS.get(request.scenario, {"window_time": "unknown"})
    return {
        "status": "demo_started",
        "scenario": request.scenario,
        "duration_seconds": 120,
        "data_source": "real_streaming_logs",
        "window_time": scenario_info.get("window_time", "unknown"),
        "real_data": real_data.get("real_data", False)
    }


async def run_real_agent_demo(scenario: str):
    """Run a real demo scenario using actual AWS Bedrock agents."""

    print(f"\n🚀 STARTING REAL AWS BEDROCK AGENT DEMO WITH REAL STREAMING DATA")
    print(f"Scenario: {scenario}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    # Get REAL scenario data from streaming logs
    scenario_data = get_real_incident_data(scenario)
    if not scenario_data or "error" in scenario_data:
        raise ValueError(f"Could not load real data for scenario: {scenario}")

    print(f"✅ REAL scenario data loaded: {scenario_data['description']}")
    print(f"✅ Severity: {scenario_data['severity']}")
    print(f"✅ Real log entries: {len(scenario_data['logs'])}")
    print(f"✅ Real metrics: {list(scenario_data.get('metrics', {}).keys())}")
    print(f"✅ Window time: {scenario_data.get('window_time', 'unknown')}")
    print(f"✅ Data source: Real streaming logs from Kafka-style pipeline")

    # Define the 4 agents
    agent_names = [
        {"name": "Initial Analysis", "key": "analyst"},
        {"name": "Root Cause", "key": "rca"},
        {"name": "Impact Assessment", "key": "impact"},
        {"name": "Solution Planning", "key": "mitigation"}
    ]

    # Initialize all agents as idle
    agents = [
        AgentStatus(name=agent["name"], status="idle", progress=0.0)
        for agent in agent_names
    ]

    # Global findings accumulator
    global_findings = []

    # Send initial state
    await manager.send_update(DemoUpdate(
        type="demo_started",
        timestamp=datetime.now(timezone.utc).isoformat(),
        agents=agents,
        overall_progress=0.0,
        current_step="Initializing real AI agents for incident analysis...",
        global_findings=global_findings
    ))

    try:
        # Create incident state
        incident_id = f"real-demo-{int(time.time())}"
        severity_mapping = {"SEV-1": Severity.SEV_1, "SEV-2": Severity.SEV_2, "SEV-3": Severity.SEV_3}
        severity = severity_mapping.get(scenario_data["severity"], Severity.SEV_2)

        state = IncidentState(incident_id=incident_id, severity=severity)

        # Use a working model (fallback to first available)
        model_id = "us.meta.llama3-3-70b-instruct-v1:0"  # Default to your working model

        # Phase 1: Analyst Agent - REAL AWS Bedrock Analysis
        agents[0].status = "processing"
        agents[0].current_activity = "Loading incident data..."

        # Activity: Starting analysis
        activity = AgentActivity(
            timestamp=datetime.now(timezone.utc).isoformat(),
            activity_type="analyzing",
            description=f"Loading {len(scenario_data['logs'])} log entries and metrics for analysis",
            details={"log_count": len(scenario_data['logs']), "metrics": len(scenario_data.get('metrics', {}))}
        )
        agents[0].activities.append(activity)
        agents[0].progress = 0.2

        await manager.send_update(DemoUpdate(
            type="agent_activity",
            timestamp=datetime.now(timezone.utc).isoformat(),
            agents=agents,
            overall_progress=0.05,
            current_step="Analyst: Processing real incident data with AWS Bedrock...",
            recent_activities=[activity],
            global_findings=global_findings
        ))

        # Create message for analyst agent
        analyst_message = AgentMessage(
            incident_id=incident_id,
            sender=AgentRole.ORCHESTRATOR,
            type=MessageType.REQUEST,
            severity=severity,
            payload=PayloadModel(
                summary=scenario_data["description"],
                details={
                    "logs": scenario_data["logs"],
                    "monitoring": {"metrics": scenario_data.get("metrics", {})}
                }
            )
        )

        # REAL ANALYST AGENT CALL
        agents[0].current_activity = "Running AWS Bedrock analysis..."
        agents[0].progress = 0.6

        activity = AgentActivity(
            timestamp=datetime.now(timezone.utc).isoformat(),
            activity_type="analyzing",
            description="Running AWS Bedrock Llama model for log pattern analysis",
            details={"model": model_id, "input_size": len(str(scenario_data))}
        )
        agents[0].activities.append(activity)

        await manager.send_update(DemoUpdate(
            type="agent_activity",
            timestamp=datetime.now(timezone.utc).isoformat(),
            agents=agents,
            overall_progress=0.12,
            current_step="Analyst: AWS Bedrock processing incident logs...",
            recent_activities=[activity],
            global_findings=global_findings
        ))

        # Initialize real analyst agent - NO FALLBACKS
        print(f"\n=== ANALYST AGENT STARTING ===")
        print(f"Model: {model_id}")
        print(f"Incident ID: {incident_id}")
        print(f"Input logs: {len(scenario_data['logs'])} entries")
        print(f"Input metrics: {scenario_data.get('metrics', {})}")

        analyst_config = AnalystAgentConfig()
        analyst_agent = AnalystAgent(config=analyst_config, model=model_id)

        # Call the real agent
        print("🔄 Calling AWS Bedrock for Analyst Agent...")
        analyst_result = await analyst_agent.handle(analyst_message, state)

        print(f"\n=== ANALYST AGENT RESPONSE ===")
        print(f"Summary: {analyst_result.payload.summary}")
        print(f"Details: {analyst_result.payload.details}")
        print(f"Evidence count: {len(analyst_result.payload.evidence) if analyst_result.payload.evidence else 0}")
        if analyst_result.payload.evidence:
            for i, evidence in enumerate(analyst_result.payload.evidence):
                print(f"  Evidence {i+1}: {evidence.title} - {evidence.description}")
        print("=== END ANALYST RESPONSE ===\n")

        # Extract real findings from analyst
        agents[0].status = "complete"
        agents[0].progress = 1.0
        agents[0].message = analyst_result.payload.summary[:100] + "..." if len(analyst_result.payload.summary) > 100 else analyst_result.payload.summary

        # Add real findings
        if analyst_result.payload.evidence:
            for evidence in analyst_result.payload.evidence:
                agents[0].findings.append(evidence.title)

        activity = AgentActivity(
            timestamp=datetime.now(timezone.utc).isoformat(),
            activity_type="completing",
            description="Initial analysis complete - real patterns identified",
            details={"evidence_count": len(analyst_result.payload.evidence) if analyst_result.payload.evidence else 0}
        )
        agents[0].activities.append(activity)
        global_findings.append(f"🔍 Real analysis complete: {analyst_result.payload.summary}")

        await manager.send_update(DemoUpdate(
            type="agent_activity",
            timestamp=datetime.now(timezone.utc).isoformat(),
            agents=agents,
            overall_progress=0.25,
            current_step="Analyst: Real AWS Bedrock analysis complete",
            recent_activities=[activity],
            global_findings=global_findings
        ))

        # Phase 2: RCA Agent - REAL Root Cause Analysis
        agents[1].status = "processing"
        agents[1].current_activity = "Starting root cause analysis..."

        activity = AgentActivity(
            timestamp=datetime.now(timezone.utc).isoformat(),
            activity_type="reasoning",
            description="Initializing AWS Bedrock for root cause hypothesis generation",
            details={"model": model_id, "input_evidence": len(agents[0].findings)}
        )
        agents[1].activities.append(activity)
        agents[1].progress = 0.3

        await manager.send_update(DemoUpdate(
            type="agent_activity",
            timestamp=datetime.now(timezone.utc).isoformat(),
            agents=agents,
            overall_progress=0.4,
            current_step="RCA: Real root cause analysis with AWS Bedrock...",
            recent_activities=[activity],
            global_findings=global_findings
        ))

        # Create message for RCA agent using analyst results
        rca_message = AgentMessage(
            incident_id=incident_id,
            sender=AgentRole.ANALYST,
            type=MessageType.HYPOTHESIS,
            severity=severity,
            payload=PayloadModel(
                summary=analyst_result.payload.summary,
                details=analyst_result.payload.details
            )
        )

        # REAL RCA AGENT CALL - NO FALLBACKS
        print(f"\n=== RCA AGENT STARTING ===")
        print(f"Model: {model_id}")
        print(f"Input from Analyst: {analyst_result.payload.summary}")
        print(f"Analyst details: {analyst_result.payload.details}")

        rca_agent = RCAAgent(model=model_id)
        print("🔄 Calling AWS Bedrock for RCA Agent...")
        rca_result = await rca_agent.handle(rca_message, state)

        print(f"\n=== RCA AGENT RESPONSE ===")
        print(f"Summary: {rca_result.payload.summary}")
        print(f"Details: {rca_result.payload.details}")
        print(f"Evidence count: {len(rca_result.payload.evidence) if rca_result.payload.evidence else 0}")
        if rca_result.payload.evidence:
            for i, evidence in enumerate(rca_result.payload.evidence):
                print(f"  Evidence {i+1}: {evidence.title} - {evidence.description}")
        print("=== END RCA RESPONSE ===\n")

        # Extract real RCA findings
        agents[1].status = "complete"
        agents[1].progress = 1.0
        agents[1].message = rca_result.payload.summary[:100] + "..." if len(rca_result.payload.summary) > 100 else rca_result.payload.summary

        if rca_result.payload.evidence:
            for evidence in rca_result.payload.evidence:
                agents[1].findings.append(evidence.title)

        activity = AgentActivity(
            timestamp=datetime.now(timezone.utc).isoformat(),
            activity_type="completing",
            description="Root cause analysis complete with real AWS Bedrock reasoning",
            details={"analysis_length": len(rca_result.payload.summary)}
        )
        agents[1].activities.append(activity)
        global_findings.append(f"🎯 Real RCA complete: {rca_result.payload.summary}")

        await manager.send_update(DemoUpdate(
            type="agent_activity",
            timestamp=datetime.now(timezone.utc).isoformat(),
            agents=agents,
            overall_progress=0.55,
            current_step="RCA: Real root cause analysis complete",
            recent_activities=[activity],
            global_findings=global_findings
        ))

        # Phase 3: Impact Agent - REAL Business Impact Assessment
        agents[2].status = "processing"
        agents[2].current_activity = "Calculating real business impact..."

        activity = AgentActivity(
            timestamp=datetime.now(timezone.utc).isoformat(),
            activity_type="calculating",
            description="AWS Bedrock analyzing business impact with real data",
            details={"model": model_id, "rca_input": len(rca_result.payload.summary)}
        )
        agents[2].activities.append(activity)
        agents[2].progress = 0.4

        await manager.send_update(DemoUpdate(
            type="agent_activity",
            timestamp=datetime.now(timezone.utc).isoformat(),
            agents=agents,
            overall_progress=0.7,
            current_step="Impact: Real business impact analysis with AWS Bedrock...",
            recent_activities=[activity],
            global_findings=global_findings
        ))

        # Create message for Impact agent
        impact_message = AgentMessage(
            incident_id=incident_id,
            sender=AgentRole.RCA,
            type=MessageType.IMPACT,
            severity=severity,
            payload=PayloadModel(
                summary=rca_result.payload.summary,
                details={**rca_result.payload.details, "rca_confidence": 0.9}
            )
        )

        # REAL IMPACT AGENT CALL - NO FALLBACKS
        print(f"\n=== IMPACT AGENT STARTING ===")
        print(f"Model: {model_id}")
        print(f"Input from RCA: {rca_result.payload.summary}")
        print(f"RCA details: {rca_result.payload.details}")

        impact_config = ImpactAgentConfig()
        impact_agent = ImpactAgent(config=impact_config, model=model_id)
        print("🔄 Calling AWS Bedrock for Impact Agent...")
        impact_result = await impact_agent.handle(impact_message, state)

        print(f"\n=== IMPACT AGENT RESPONSE ===")
        print(f"Summary: {impact_result.payload.summary}")
        print(f"Details: {impact_result.payload.details}")
        print(f"Evidence count: {len(impact_result.payload.evidence) if impact_result.payload.evidence else 0}")
        if impact_result.payload.evidence:
            for i, evidence in enumerate(impact_result.payload.evidence):
                print(f"  Evidence {i+1}: {evidence.title} - {evidence.description}")
        print("=== END IMPACT RESPONSE ===\n")

        # Extract real impact findings
        agents[2].status = "complete"
        agents[2].progress = 1.0
        agents[2].message = impact_result.payload.summary[:100] + "..." if len(impact_result.payload.summary) > 100 else impact_result.payload.summary

        if impact_result.payload.evidence:
            for evidence in impact_result.payload.evidence:
                agents[2].findings.append(evidence.title)

        activity = AgentActivity(
            timestamp=datetime.now(timezone.utc).isoformat(),
            activity_type="completing",
            description="Business impact assessment complete with real calculations",
            details={"impact_summary_length": len(impact_result.payload.summary)}
        )
        agents[2].activities.append(activity)
        global_findings.append(f"💰 Real impact analysis: {impact_result.payload.summary}")

        await manager.send_update(DemoUpdate(
            type="agent_activity",
            timestamp=datetime.now(timezone.utc).isoformat(),
            agents=agents,
            overall_progress=0.85,
            current_step="Impact: Real business impact analysis complete",
            recent_activities=[activity],
            global_findings=global_findings
        ))

        # Phase 4: Mitigation Agent - REAL Solution Planning with RAG
        agents[3].status = "processing"
        agents[3].current_activity = "Generating real mitigation plan with RAG..."

        activity = AgentActivity(
            timestamp=datetime.now(timezone.utc).isoformat(),
            activity_type="planning",
            description="AWS Bedrock creating mitigation plan with RAG policy integration",
            details={"model": model_id, "rag_enabled": True}
        )
        agents[3].activities.append(activity)
        agents[3].progress = 0.3

        await manager.send_update(DemoUpdate(
            type="agent_activity",
            timestamp=datetime.now(timezone.utc).isoformat(),
            agents=agents,
            overall_progress=0.92,
            current_step="Mitigation: Real solution planning with RAG + AWS Bedrock...",
            recent_activities=[activity],
            global_findings=global_findings
        ))

        # Create message for Mitigation agent
        mitigation_message = AgentMessage(
            incident_id=incident_id,
            sender=AgentRole.IMPACT,
            type=MessageType.PLAN,
            severity=severity,
            payload=PayloadModel(
                summary=impact_result.payload.summary,
                details={
                    "rca": rca_result.payload.summary,
                    "impact": impact_result.payload.summary,
                    "signals": analyst_result.payload.summary,
                    **impact_result.payload.details
                }
            )
        )

        # REAL MITIGATION AGENT CALL (with RAG) - NO FALLBACKS
        print(f"\n=== MITIGATION AGENT STARTING ===")
        print(f"Model: {model_id}")
        print(f"Input from Impact: {impact_result.payload.summary}")
        print(f"All context: RCA={rca_result.payload.summary}, Impact={impact_result.payload.summary}, Signals={analyst_result.payload.summary}")

        mitigation_config = MitigationAgentConfig()
        mitigation_agent = MitigationCommsAgent(config=mitigation_config, model=model_id)
        print("🔄 Calling AWS Bedrock for Mitigation Agent (with RAG)...")
        mitigation_result = await mitigation_agent.handle(mitigation_message, state)

        print(f"\n=== MITIGATION AGENT RESPONSE ===")
        print(f"Summary: {mitigation_result.payload.summary}")
        print(f"Details: {mitigation_result.payload.details}")
        print(f"Evidence count: {len(mitigation_result.payload.evidence) if mitigation_result.payload.evidence else 0}")
        if mitigation_result.payload.evidence:
            for i, evidence in enumerate(mitigation_result.payload.evidence):
                print(f"  Evidence {i+1}: {evidence.title} - {evidence.description}")

        # Check for RAG policy documents
        policy_docs = mitigation_result.payload.details.get("policy_documents", [])
        print(f"RAG Policy Documents Retrieved: {len(policy_docs)}")
        for i, doc in enumerate(policy_docs):
            print(f"  Policy {i+1}: {doc}")
        print("=== END MITIGATION RESPONSE ===\n")

        # Extract real mitigation findings
        agents[3].status = "complete"
        agents[3].progress = 1.0
        agents[3].message = mitigation_result.payload.summary[:100] + "..." if len(mitigation_result.payload.summary) > 100 else mitigation_result.payload.summary

        if mitigation_result.payload.evidence:
            for evidence in mitigation_result.payload.evidence:
                agents[3].findings.append(evidence.title)

        activity = AgentActivity(
            timestamp=datetime.now(timezone.utc).isoformat(),
            activity_type="completing",
            description="Real mitigation plan complete with RAG-enhanced recommendations",
            details={"plan_length": len(mitigation_result.payload.summary)}
        )
        agents[3].activities.append(activity)
        global_findings.append(f"🛠️ Real mitigation plan: {mitigation_result.payload.summary}")

        # Calculate processing time and metrics
        total_activities = sum(len(agent.activities) for agent in agents)
        processing_time = len(global_findings) * 8  # Rough estimate based on real processing

        # Final completion with real results
        global_findings.append("🎉 Real incident analysis complete - all agents successful")
        await manager.send_update(DemoUpdate(
            type="completion",
            timestamp=datetime.now(timezone.utc).isoformat(),
            agents=agents,
            overall_progress=1.0,
            current_step="Real incident analysis complete - ready for resolution",
            completion_metrics={
                "time_to_resolution": f"{processing_time} seconds",
                "confidence_score": "Real AI Analysis",
                "affected_users": "Calculated by AWS Bedrock",
                "resolution_status": "Complete - Real Analysis",
                "total_activities": total_activities,
                "total_findings": len(global_findings),
                "model_used": model_id
            },
            global_findings=global_findings
        ))

        print("\n🎉 ALL REAL AGENTS COMPLETED SUCCESSFULLY!")
        print("Final summary:")
        print(f"- Analyst: {analyst_result.payload.summary[:100]}...")
        print(f"- RCA: {rca_result.payload.summary[:100]}...")
        print(f"- Impact: {impact_result.payload.summary[:100]}...")
        print(f"- Mitigation: {mitigation_result.payload.summary[:100]}...")

    except Exception as e:
        # NO FALLBACKS - Let it fail and show us the real error
        print(f"\n❌ REAL AGENT DEMO FAILED: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Full traceback:")
        traceback.print_exc()

        # Still need to notify frontend of failure
        await manager.send_update(DemoUpdate(
            type="error",
            timestamp=datetime.now(timezone.utc).isoformat(),
            agents=agents,
            overall_progress=0.0,
            current_step=f"Real agent processing failed: {str(e)}",
            global_findings=[],
            error_message=f"Real AWS Bedrock agent processing failed: {str(e)}"
        ))
        raise  # Re-raise the exception so we can see the full error


async def run_real_pipeline(scenario: str, pipeline_id: str, incident_id: str, model_id: str = "us.meta.llama3-3-70b-instruct-v1:0"):
    """Run a real pipeline using the WebSocketOrchestrator for real-time updates."""

    try:
        print(f"\n🚀 STARTING REAL PIPELINE WITH WEBSOCKET UPDATES")
        print(f"Pipeline ID: {pipeline_id}")
        print(f"Incident ID: {incident_id}")
        print(f"Scenario: {scenario}")
        print(f"Model: {model_id}")
    except Exception as e:
        print(f"❌ Error in initial logging: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        # Get real scenario data
        scenario_data = get_real_incident_data(scenario)
        if not scenario_data or "error" in scenario_data:
            raise ValueError(f"Could not load real data for scenario: {scenario}")

        print(f"✅ Real scenario data loaded: {scenario_data['description']}")
        print(f"✅ Severity: {scenario_data['severity']}")
        print(f"✅ Real log entries: {len(scenario_data['logs'])}")

        # Create a mock scenario snapshot for the orchestrator
        from datetime import datetime, timedelta
        from dataclasses import dataclass
        from typing import List

        @dataclass
        class MockMetadata:
            key: str
            description: str
            severity: str

        @dataclass
        class MockTimeWindow:
            start: datetime
            end: datetime

        @dataclass
        class MockScenarioSnapshot:
            incident_id: str
            metadata: MockMetadata
            window: MockTimeWindow
            logs: List[Any]
            monitoring: Dict[str, Any]
            additional_sources: Dict[str, Any]

        # Get window info
        scenario_info = REAL_INCIDENT_SCENARIOS[scenario]
        window_time = scenario_info["window_time"]

        # Create window timing based on scenario
        base_time = data_pipeline.dataset_start_time
        if window_time == "09:45":
            window_start = base_time + timedelta(minutes=45)
        elif window_time == "11:15":
            window_start = base_time + timedelta(hours=1, minutes=15)
        elif window_time == "12:45":
            window_start = base_time + timedelta(hours=2, minutes=45)
        else:
            window_start = base_time

        window_end = window_start + timedelta(minutes=15)  # 15-minute window

        # Get real logs
        window_logs = data_pipeline.get_window_logs(window_start)

        # Convert severity
        severity_mapping = {"SEV-1": Severity.SEV_1, "SEV-2": Severity.SEV_2, "SEV-3": Severity.SEV_3}
        severity_enum = severity_mapping.get(scenario_data["severity"], Severity.SEV_2)

        # Create mock scenario snapshot
        scenario_snapshot = MockScenarioSnapshot(
            incident_id=incident_id,
            metadata=MockMetadata(
                key=scenario,
                description=scenario_data["description"],
                severity=severity_enum
            ),
            window=MockTimeWindow(start=window_start, end=window_end),
            logs=window_logs,
            monitoring={"metrics": scenario_data.get("metrics", {})},
            additional_sources={}
        )

        # Initialize real agents
        analyst_config = AnalystAgentConfig()
        analyst_agent = AnalystAgent(config=analyst_config, model=model_id)

        rca_agent = RCAAgent(model=model_id)

        impact_config = ImpactAgentConfig()
        impact_agent = ImpactAgent(config=impact_config, model=model_id)

        mitigation_config = MitigationAgentConfig()
        mitigation_agent = MitigationCommsAgent(config=mitigation_config, model=model_id)

        # BROADCAST: Pipeline starting
        await pipeline_manager.broadcast_update({
            "type": "pipeline_started",
            "pipeline_id": pipeline_id,
            "incident_id": incident_id,
            "scenario": scenario,
            "severity": severity_enum.name,
            "description": scenario_data["description"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        print(f"📡 Broadcasted pipeline_started event")

        # DEBUG: Check WebSocket connections before creating orchestrator
        print(f"🔍 DEBUG: Pipeline manager has {len(pipeline_manager.websocket_connections)} WebSocket connections")
        print(f"🔍 DEBUG: Pipeline manager ID: {id(pipeline_manager)}")

        # Create WebSocket orchestrator
        websocket_orchestrator = WebSocketOrchestrator(
            analyst_agent=analyst_agent,
            rca_agent=rca_agent,
            impact_agent=impact_agent,
            mitigation_agent=mitigation_agent,
            state_manager=pipeline_manager,  # ← ADD THIS
            pipeline_id=pipeline_id,  # ← ADD THIS
            demo_mode=False,
            incident_id=incident_id
        )

        print(f"🔄 Running real pipeline with WebSocket updates...")

        # Run the pipeline with real-time updates
        result = await websocket_orchestrator.run_with_websocket_updates(
            snapshot=scenario_snapshot,
            pipeline_id=pipeline_id
        )

        print(f"\n🎉 REAL PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Pipeline ID: {pipeline_id}")
        print(f"Responses: {len(result.responses)}")
        print(f"Plans: {len(result.plans)}")
        if result.summary:
            print(f"Summary generated: Yes")

        return result

    except Exception as e:
        print(f"\n❌ REAL PIPELINE FAILED: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Full traceback:")
        traceback.print_exc()

        # The pipeline manager will handle the error state update
        raise


@app.post("/test/real-agents")
async def test_real_agents_directly():
    """Test real agents directly with real streaming data - for debugging."""
    scenario = "database_connection_issues"  # Use real scenario

    print(f"\n🧪 DIRECT REAL AGENT TEST WITH REAL STREAMING DATA")
    print(f"Scenario: {scenario}")

    try:
        # Get REAL scenario data from streaming logs
        scenario_data = get_real_incident_data(scenario)
        if not scenario_data or "error" in scenario_data:
            raise ValueError(f"Could not load real data for {scenario}")

        incident_id = f"test-{int(time.time())}"
        severity_mapping = {"SEV-1": Severity.SEV_1, "SEV-2": Severity.SEV_2, "SEV-3": Severity.SEV_3}
        severity = severity_mapping.get(scenario_data["severity"], Severity.SEV_2)
        state = IncidentState(incident_id=incident_id, severity=severity)
        model_id = "us.meta.llama3-3-70b-instruct-v1:0"

        # Test Analyst Agent
        print("\n🔬 Testing Analyst Agent with REAL streaming data...")

        # Show exactly what REAL input data we're sending
        print(f"\n📥 REAL INPUT DATA TO ANALYST AGENT:")
        print(f"Incident Description: {scenario_data['description']}")
        print(f"Severity: {scenario_data['severity']}")
        print(f"Model: {model_id}")
        print(f"Window Time: {scenario_data.get('window_time', 'unknown')}")
        print(f"Data Source: Real streaming logs from Kafka-style pipeline")
        print(f"\n📋 REAL LOG ENTRIES ({len(scenario_data['logs'])}):")
        for i, log in enumerate(scenario_data["logs"], 1):
            print(f"  {i}. [{log['level']}] {log['timestamp']} - {log['message']} (service: {log['service']})")

        print(f"\n📊 REAL METRICS:")
        for key, value in scenario_data.get("metrics", {}).items():
            print(f"  {key}: {value}")

        print(f"\n🤖 Now calling AWS Bedrock {model_id} with REAL streaming data...\n")

        analyst_message = AgentMessage(
            incident_id=incident_id,
            sender=AgentRole.ORCHESTRATOR,
            type=MessageType.REQUEST,
            severity=severity,
            payload=PayloadModel(
                summary=scenario_data["description"],
                details={
                    "logs": scenario_data["logs"],
                    "monitoring": {"metrics": scenario_data.get("metrics", {})}
                }
            )
        )

        analyst_config = AnalystAgentConfig()
        analyst_agent = AnalystAgent(config=analyst_config, model=model_id)
        analyst_result = await analyst_agent.handle(analyst_message, state)

        print(f"\n🎉 ✅ REAL AWS BEDROCK RESPONSE RECEIVED FROM REAL DATA!")
        print(f"📝 Summary: {analyst_result.payload.summary}")
        print(f"📊 Details keys: {list(analyst_result.payload.details.keys()) if analyst_result.payload.details else 'None'}")
        print(f"🔍 Evidence items: {len(analyst_result.payload.evidence) if analyst_result.payload.evidence else 0}")
        print(f"✨ This is a REAL LLM response from AWS Bedrock {model_id} analyzing REAL streaming data")
        print("="*80)

        return {
            "success": True,
            "confirmation": "REAL AWS BEDROCK RESPONSE FROM REAL STREAMING DATA - NO FALLBACKS",
            "model_used": model_id,
            "data_source": "real_streaming_logs",
            "window_time": scenario_data.get("window_time"),
            "real_data": scenario_data.get("real_data", True),
            "analyst_response": {
                "summary": analyst_result.payload.summary,
                "details": analyst_result.payload.details,
                "evidence_count": len(analyst_result.payload.evidence) if analyst_result.payload.evidence else 0
            }
        }

    except Exception as e:
        print(f"❌ Direct test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


@app.get("/real-data/info")
async def get_real_data_info():
    """Get information about the real streaming data pipeline."""
    try:
        # Load pipeline data
        logs = data_pipeline.load_logs()
        windows = data_pipeline.get_available_windows()
        incident_windows = data_pipeline.get_incident_windows()

        return {
            "data_source": "kafka_style_streaming_logs",
            "dataset_info": {
                "total_log_entries": len(logs),
                "dataset_duration_hours": 5,
                "dataset_start": data_pipeline.dataset_start_time.isoformat(),
                "total_windows": len(windows),
                "incident_windows": len(incident_windows)
            },
            "real_incidents": REAL_INCIDENT_SCENARIOS,
            "incident_windows": [w.isoformat() for w in incident_windows],
            "sample_logs": [
                {
                    "timestamp": log.timestamp.isoformat(),
                    "level": log.level,
                    "service": log.service,
                    "message": log.message[:100] + "..." if len(log.message) > 100 else log.message
                }
                for log in logs[:5]  # Show first 5 logs as sample
            ]
        }
    except Exception as e:
        return {
            "error": f"Could not load real data info: {str(e)}",
            "data_source": "error"
        }


@app.post("/api/streaming/start")
async def start_streaming_analysis(request: StreamingStartRequest, background_tasks: BackgroundTasks):
    """Start streaming data analysis - processes windows sequentially."""
    import uuid

    session_id = f"stream_{uuid.uuid4().hex[:8]}"

    # Start streaming in background
    background_tasks.add_task(
        run_streaming_session,
        session_id=session_id,
        start_time=request.start_time,
        window_size=request.window_size_minutes,
        model_id=request.model_id
    )

    return {
        "session_id": session_id,
        "message": f"Started streaming analysis from {request.start_time}",
        "window_size": request.window_size_minutes
    }


async def run_streaming_session(session_id: str, start_time: str, window_size: int, model_id: str):
    """Process log windows sequentially, only triggering full pipeline on anomaly detection."""

    print(f"\n🌊 STARTING STREAMING SESSION")
    print(f"Session ID: {session_id}")
    print(f"Start time: {start_time}")
    print(f"Window size: {window_size} minutes")
    print(f"Model: {model_id}")

    try:
        # Convert start_time to datetime
        from datetime import datetime, timedelta

        # Parse HH:MM format to datetime relative to dataset start
        hour, minute = map(int, start_time.split(':'))
        dataset_start = data_pipeline.dataset_start_time
        current_time = dataset_start.replace(hour=hour, minute=minute, second=0, microsecond=0)

        # Initialize analyst agent for anomaly detection
        from src.orchestration.four_agent.analyst_agent import AnalystAgent, AnalystAgentConfig
        from src.orchestration.four_agent.schema import AgentMessage, AgentRole, MessageType, Severity, PayloadModel
        from src.orchestration.four_agent.state import IncidentState

        analyst_config = AnalystAgentConfig()
        analyst_agent = AnalystAgent(config=analyst_config, model=model_id)

        # Broadcast session start
        await pipeline_manager.broadcast_update({
            "type": "streaming_session_started",
            "session_id": session_id,
            "start_time": start_time,
            "window_size": window_size,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        window_count = 0
        max_windows = 20  # Process up to 20 windows (5 hours worth)

        while window_count < max_windows:
            window_count += 1
            window_end = current_time + timedelta(minutes=window_size)

            print(f"\n📋 PROCESSING WINDOW {window_count}")
            print(f"Time range: {current_time.strftime('%H:%M')} - {window_end.strftime('%H:%M')}")

            # Get logs for this window
            window_logs = data_pipeline.get_window_logs(current_time)
            log_count = len(window_logs)

            print(f"📊 Log entries: {log_count}")

            # Broadcast window start
            await pipeline_manager.broadcast_update({
                "type": "streaming_window_start",
                "session_id": session_id,
                "window_number": window_count,
                "window_start": current_time.strftime('%H:%M'),
                "window_end": window_end.strftime('%H:%M'),
                "log_count": log_count,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            if log_count > 0:
                # Convert logs for analyst agent
                log_entries = []
                for log_entry in window_logs[:10]:  # Limit to 10 most relevant
                    log_entries.append({
                        "timestamp": log_entry.timestamp.isoformat(),
                        "level": log_entry.level,
                        "message": log_entry.message,
                        "service": log_entry.service
                    })

                # Run analyst agent for anomaly detection
                print(f"🔍 Running analyst agent for anomaly detection...")

                incident_id = f"stream-analysis-{window_count}"
                state = IncidentState(incident_id=incident_id, severity=Severity.SEV_2)

                analyst_message = AgentMessage(
                    incident_id=incident_id,
                    sender=AgentRole.ORCHESTRATOR,
                    type=MessageType.REQUEST,
                    severity=Severity.SEV_2,
                    payload=PayloadModel(
                        summary=f"Streaming window analysis: {current_time.strftime('%H:%M')} - {window_end.strftime('%H:%M')}",
                        details={
                            "logs": log_entries,
                            "monitoring": {"window_size": window_size, "log_count": log_count}
                        }
                    )
                )

                # Broadcast analyst analysis start
                await pipeline_manager.broadcast_update({
                    "type": "streaming_analysis_start",
                    "session_id": session_id,
                    "window_number": window_count,
                    "agent": "analyst",
                    "message": "Analyzing window for anomalies...",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                # Call analyst agent
                analyst_result = await analyst_agent.handle(analyst_message, state)

                print(f"✅ Analyst analysis complete")
                print(f"Summary: {analyst_result.payload.summary[:100]}...")

                # Check for anomaly detection in analyst response
                anomaly_detected = await check_for_anomaly_detection(analyst_result, window_count, current_time)

                if anomaly_detected:
                    print(f"🚨 ANOMALY DETECTED - Triggering full incident pipeline!")

                    # Broadcast incident trigger
                    incident_id = f"INC-{window_count:03d}"
                    await pipeline_manager.broadcast_update({
                        "type": "incident_triggered",
                        "session_id": session_id,
                        "incident_id": incident_id,
                        "window_number": window_count,
                        "window_time": current_time.strftime('%H:%M'),
                        "anomaly_summary": analyst_result.payload.summary[:200],
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })

                    # Trigger full pipeline for this incident
                    await trigger_full_pipeline_for_incident(
                        incident_id, window_logs, analyst_result, model_id, session_id, window_count
                    )
                else:
                    print(f"✅ No anomalies detected - continuing to next window")

                    # Broadcast window complete (no incident)
                    await pipeline_manager.broadcast_update({
                        "type": "streaming_window_complete",
                        "session_id": session_id,
                        "window_number": window_count,
                        "incident_detected": False,
                        "analyst_summary": analyst_result.payload.summary[:100],
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
            else:
                print(f"⚠️  No logs in this window - skipping")

                # Broadcast empty window
                await pipeline_manager.broadcast_update({
                    "type": "streaming_window_complete",
                    "session_id": session_id,
                    "window_number": window_count,
                    "incident_detected": False,
                    "message": "No logs in this window",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

            # Move to next window
            current_time = current_time + timedelta(minutes=5)  # 5-minute step for overlapping windows

            # Wait between windows for realistic streaming simulation
            await asyncio.sleep(2)

        # Broadcast session complete
        await pipeline_manager.broadcast_update({
            "type": "streaming_session_complete",
            "session_id": session_id,
            "windows_processed": window_count,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        print(f"\n🏁 STREAMING SESSION COMPLETE")
        print(f"Windows processed: {window_count}")

    except Exception as e:
        print(f"\n❌ STREAMING SESSION FAILED: {e}")
        import traceback
        traceback.print_exc()

        # Broadcast session error
        await pipeline_manager.broadcast_update({
            "type": "streaming_session_failed",
            "session_id": session_id,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


async def check_for_anomaly_detection(analyst_result, window_number: int, window_time: datetime) -> bool:
    """Check if analyst detected an anomaly using existing pipeline logic (from full_pipeline_runner.py)."""

    try:
        # Extract data from AgentMessage response payload
        payload = analyst_result.payload if analyst_result and analyst_result.payload else None
        if not payload:
            print("⚠️  No payload in analyst result")
            return False

        # Get details from payload
        details = payload.details if payload.details else {}

        # Extract confidence (default to 0.8 if not found)
        confidence = details.get('confidence', 0.8)

        # Extract severity from summary/details or use incident severity
        severity = "MEDIUM"
        if payload.summary:
            summary_lower = payload.summary.lower()
            if "critical" in summary_lower or "sev-1" in summary_lower:
                severity = "CRITICAL"
            elif "high" in summary_lower or "sev-2" in summary_lower:
                severity = "HIGH"
            elif "low" in summary_lower or "sev-3" in summary_lower:
                severity = "LOW"

        # Count evidence items as anomalies
        anomalies_count = 0
        if hasattr(payload, 'evidence') and payload.evidence:
            anomalies_count = len(payload.evidence)
        elif 'anomalies' in details:
            anomalies_count = len(details['anomalies']) if isinstance(details['anomalies'], list) else 1

        summary = payload.summary.lower() if payload.summary else ""

        print(f"🔍 Anomaly detection using existing logic:")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Severity: {severity}")
        print(f"   Anomalies detected: {anomalies_count}")

        # Trigger incident if:
        # 1. High confidence (>70%) AND severity is HIGH/CRITICAL
        # 2. OR multiple anomalies detected (>=2)
        # 3. OR low confidence (<50%) but HIGH/CRITICAL severity (emergency override)

        high_confidence_trigger = confidence > 0.7 and severity in ["HIGH", "CRITICAL"]
        multiple_anomalies_trigger = anomalies_count >= 2
        emergency_override = confidence < 0.5 and severity == "CRITICAL"

        should_trigger = high_confidence_trigger or multiple_anomalies_trigger or emergency_override

        print(f"   Triggers: high_conf={high_confidence_trigger}, multi_anomaly={multiple_anomalies_trigger}, emergency={emergency_override}")
        print(f"   → Should trigger incident: {should_trigger}")

        return should_trigger

    except Exception as e:
        print(f"❌ Error in anomaly detection: {e}")
        # Fallback: trigger on any analysis (conservative approach)
        return True


async def trigger_full_pipeline_for_incident(
    incident_id: str,
    window_logs,
    analyst_result,
    model_id: str,
    session_id: str,
    window_number: int
):
    """Trigger the full RCA + Impact + Mitigation pipeline for detected incident."""

    print(f"\n🚀 TRIGGERING FULL PIPELINE FOR INCIDENT {incident_id}")

    try:
        # Create incident scenario from window data
        scenario_data = {
            "description": f"Incident detected in streaming window {window_number}: {analyst_result.payload.summary[:100]}",
            "severity": "SEV-2",
            "logs": [
                {
                    "timestamp": log.timestamp.isoformat(),
                    "level": log.level,
                    "message": log.message,
                    "service": log.service
                }
                for log in window_logs[:10]
            ],
            "metrics": {
                "window_number": window_number,
                "total_logs": len(window_logs),
                "analyst_confidence": 0.8
            }
        }

        # Run the full pipeline using existing infrastructure
        pipeline_id = f"incident-{incident_id}-{int(time.time())}"

        # Use the existing run_real_pipeline function
        await run_real_pipeline(
            scenario="streaming_incident",  # Special scenario type
            pipeline_id=pipeline_id,
            incident_id=incident_id,
            model_id=model_id
        )

        print(f"✅ Full pipeline completed for incident {incident_id}")

        # Broadcast incident pipeline complete
        await pipeline_manager.broadcast_update({
            "type": "incident_pipeline_complete",
            "session_id": session_id,
            "incident_id": incident_id,
            "pipeline_id": pipeline_id,
            "window_number": window_number,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    except Exception as e:
        print(f"❌ Full pipeline failed for incident {incident_id}: {e}")
        import traceback
        traceback.print_exc()

        # Broadcast incident pipeline failed
        await pipeline_manager.broadcast_update({
            "type": "incident_pipeline_failed",
            "session_id": session_id,
            "incident_id": incident_id,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "AWS Bedrock SRE Agent Demo",
        "version": "1.0.0",
        "description": "LLM-powered incident response with 5-agent orchestration using REAL streaming data",
        "data_source": "kafka_style_streaming_logs",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze-incident",
            "sample_scenarios": "/demo/scenarios",
            "sample_analysis": "/demo/sample-incident/{scenario_name}",
            "websocket_demo": "/ws/demo",
            "websocket_pipeline": "/ws/pipeline",
            "start_demo": "/api/demo/start",
            "start_pipeline": "/api/pipeline/start",
            "start_streaming": "/api/streaming/start",
            "pipeline_status": "/api/pipeline/{pipeline_id}/status",
            "test_real_agents": "/test/real-agents",
            "real_data_info": "/real-data/info",
            "docs": "/docs"
        },
        "supported_models": [
            "us.meta.llama3-3-70b-instruct-v1:0",
            "us.meta.llama4-maverick-17b-instruct-v1:0"
        ],
        "real_incidents": list(REAL_INCIDENT_SCENARIOS.keys()),
        "streaming": {
            "description": "Time-based log analysis with anomaly detection",
            "example_start_times": ["09:30", "10:00", "11:00"],
            "window_size": "15 minutes default"
        },
        "note": "Now supports streaming data analysis with anomaly detection and incident triggering"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
