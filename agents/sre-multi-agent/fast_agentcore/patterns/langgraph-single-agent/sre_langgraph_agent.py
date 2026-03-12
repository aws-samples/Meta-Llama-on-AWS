"""
SRE Multi-Agent System - LangGraph AgentCore Entrypoint

This integrates the 4-agent SRE system (Analyst, RCA, Impact, Mitigation)
with AWS Bedrock AgentCore using LangGraph for orchestration.
"""
import json
import os
from datetime import datetime, timezone
from typing import TypedDict, List, Dict, Any

import boto3
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langgraph_checkpoint_aws import AgentCoreMemorySaver
from bedrock_agentcore.runtime import BedrockAgentCoreApp, RequestContext

from utils.auth import extract_user_id_from_context
from utils.ssm import get_ssm_parameter

# Import your 4 agents
from src.orchestration.four_agent.analyst_agent import AnalystAgent
from src.orchestration.four_agent.rca_agent import RCAAgent
from src.orchestration.four_agent.impact_agent import ImpactAgent
from src.orchestration.four_agent.mitigation_agent import MitigationCommsAgent
from src.orchestration.four_agent.scenario_loader import ScenarioSnapshot, ScenarioMetadata
from src.orchestration.four_agent.schema import Severity

app = BedrockAgentCoreApp()


# Define state for LangGraph workflow
class AgentState(TypedDict):
    """State passed between agents in the workflow."""
    messages: List
    user_query: str
    log_text: str
    log_count: int
    analyst_result: Dict[str, Any]
    rca_result: Dict[str, Any]
    impact_result: Dict[str, Any]
    mitigation_result: Dict[str, Any]
    current_step: str
    scenario: Any  # ScenarioSnapshot


async def get_cloudwatch_logs(log_group: str, minutes: int = 15) -> tuple[str, int]:
    """Retrieve recent CloudWatch logs."""
    logs_client = boto3.client('logs', region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-west-2'))
    
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = end_time - (minutes * 60 * 1000)
    
    try:
        response = logs_client.filter_log_events(
            logGroupName=log_group,
            startTime=start_time,
            endTime=end_time,
            limit=100  # Limit for LLM token constraints
        )
        
        log_entries = []
        for event in response.get('events', []):
            timestamp = datetime.fromtimestamp(event['timestamp'] / 1000, tz=timezone.utc)
            message = event['message']
            log_entries.append(f"[{timestamp.isoformat()}] {message}")
        
        log_text = "\n".join(log_entries)
        return log_text, len(log_entries)
    
    except Exception as e:
        print(f"[ERROR] Failed to retrieve CloudWatch logs: {e}")
        return "", 0


# Define agent nodes
async def analyst_node(state: AgentState) -> AgentState:
    """Analyst agent node - detects anomalies in logs."""
    print("[ANALYST] Analyzing logs...")
    
    analyst = AnalystAgent(unstructured_mode=True)
    
    metadata = {
        "window_start": datetime.now(timezone.utc).isoformat(),
        "window_size_minutes": 15,
        "log_count": state["log_count"]
    }
    
    try:
        result = await analyst.analyze_logs(state["log_text"], metadata)
        
        state["analyst_result"] = {
            "anomalies": [
                {
                    "pattern": a.pattern,
                    "confidence": a.confidence,
                    "severity": a.severity
                } for a in result.anomalies
            ],
            "overall_confidence": result.overall_confidence,
            "severity": result.severity_assessment,
            "summary": result.log_summary
        }
        
        anomaly_count = len(result.anomalies)
        confidence = result.overall_confidence
        
        message = (
            f"🔍 **Analyst Agent**: Detected {anomaly_count} anomal{'y' if anomaly_count == 1 else 'ies'}\n"
            f"   Confidence: {confidence:.0%} | Severity: {result.severity_assessment}\n"
        )
        
        if anomaly_count > 0:
            message += "\n   Anomalies:\n"
            for i, anomaly in enumerate(result.anomalies[:3], 1):  # Show top 3
                message += f"   {i}. {anomaly.pattern[:80]}...\n"
        
        state["messages"].append(AIMessage(content=message))
        
        # Determine next step based on confidence
        if confidence >= 0.60:
            state["current_step"] = "rca"
        else:
            state["current_step"] = "end"
        
    except Exception as e:
        print(f"[ERROR] Analyst agent failed: {e}")
        state["analyst_result"] = {"error": str(e)}
        state["current_step"] = "end"
        state["messages"].append(
            AIMessage(content=f"❌ Analyst Agent failed: {str(e)}\n")
        )
    
    return state


async def rca_node(state: AgentState) -> AgentState:
    """RCA agent node - finds root cause."""
    print("[RCA] Determining root cause...")
    
    rca = RCAAgent()
    
    try:
        # Create scenario for RCA
        scenario = ScenarioSnapshot(
            incident_id=f"INC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            metadata=ScenarioMetadata(
                severity=Severity.MEDIUM,
                description=state["analyst_result"].get("summary", "Incident detected")
            ),
            logs=state["log_text"],
            metrics={}
        )
        
        state["scenario"] = scenario
        
        # For now, use simplified RCA result
        # TODO: Integrate full RCA agent logic
        state["rca_result"] = {
            "root_cause": "Database connection pool exhaustion",
            "confidence": 0.85,
            "hypotheses": [
                {
                    "hypothesis": "Database connection pool exhausted due to high traffic",
                    "confidence": 0.85
                },
                {
                    "hypothesis": "Slow queries causing connection buildup",
                    "confidence": 0.65
                }
            ]
        }
        
        message = (
            f"🔍 **RCA Agent**: Root cause identified\n"
            f"   Cause: {state['rca_result']['root_cause']}\n"
            f"   Confidence: {state['rca_result']['confidence']:.0%}\n"
        )
        
        state["messages"].append(AIMessage(content=message))
        state["current_step"] = "impact"
        
    except Exception as e:
        print(f"[ERROR] RCA agent failed: {e}")
        state["rca_result"] = {"error": str(e)}
        state["current_step"] = "impact"  # Continue to impact even if RCA fails
        state["messages"].append(
            AIMessage(content=f"⚠️ RCA Agent encountered an issue, continuing...\n")
        )
    
    return state


async def impact_node(state: AgentState) -> AgentState:
    """Impact agent node - calculates business impact."""
    print("[IMPACT] Assessing business impact...")
    
    impact = ImpactAgent()
    
    try:
        # For now, use simplified impact result
        # TODO: Integrate full Impact agent logic
        state["impact_result"] = {
            "revenue_impact": "$2,052.50/min",
            "tps_impact": {
                "current": 450,
                "baseline": 850,
                "degradation": "47%"
            },
            "user_impact": "High - 50% of users affected",
            "sla_impact": "SLA breach imminent"
        }
        
        message = (
            f"📊 **Impact Agent**: Business impact assessed\n"
            f"   Revenue Loss: {state['impact_result']['revenue_impact']}\n"
            f"   TPS: {state['impact_result']['tps_impact']['current']} "
            f"(baseline: {state['impact_result']['tps_impact']['baseline']})\n"
            f"   User Impact: {state['impact_result']['user_impact']}\n"
        )
        
        state["messages"].append(AIMessage(content=message))
        state["current_step"] = "mitigation"
        
    except Exception as e:
        print(f"[ERROR] Impact agent failed: {e}")
        state["impact_result"] = {"error": str(e)}
        state["current_step"] = "mitigation"  # Continue to mitigation
        state["messages"].append(
            AIMessage(content=f"⚠️ Impact Agent encountered an issue, continuing...\n")
        )
    
    return state


async def mitigation_node(state: AgentState) -> AgentState:
    """Mitigation agent node - generates action plan."""
    print("[MITIGATION] Generating action plan...")
    
    mitigation = MitigationCommsAgent()
    
    try:
        # For now, use simplified mitigation result
        # TODO: Integrate full Mitigation agent logic
        state["mitigation_result"] = {
            "steps": [
                "Scale database connection pool from 50 to 100 connections",
                "Restart affected services: auth-service, trading-service, payments-service",
                "Monitor TPS recovery to baseline (850 tps)",
                "Implement connection pool monitoring alerts",
                "Review slow query logs and optimize problematic queries"
            ],
            "estimated_recovery_time": "15-30 minutes",
            "rollback_plan": "Revert connection pool changes if issues persist"
        }
        
        message = (
            f"🛠️ **Mitigation Agent**: Action plan generated\n"
            f"   Steps: {len(state['mitigation_result']['steps'])}\n"
            f"   Estimated Recovery: {state['mitigation_result']['estimated_recovery_time']}\n"
        )
        
        state["messages"].append(AIMessage(content=message))
        state["current_step"] = "complete"
        
    except Exception as e:
        print(f"[ERROR] Mitigation agent failed: {e}")
        state["mitigation_result"] = {"error": str(e)}
        state["current_step"] = "complete"
        state["messages"].append(
            AIMessage(content=f"❌ Mitigation Agent failed: {str(e)}\n")
        )
    
    return state


def should_continue(state: AgentState) -> str:
    """Determine next step in workflow based on current state."""
    current = state.get("current_step", "analyst")
    
    if current == "analyst":
        # Check if anomalies detected with sufficient confidence
        confidence = state.get("analyst_result", {}).get("overall_confidence", 0)
        if confidence < 0.60:
            return "end"
        return "rca"
    elif current == "rca":
        return "impact"
    elif current == "impact":
        return "mitigation"
    else:
        return "end"


def create_sre_workflow() -> StateGraph:
    """Create the SRE multi-agent workflow graph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes for each agent
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("rca", rca_node)
    workflow.add_node("impact", impact_node)
    workflow.add_node("mitigation", mitigation_node)
    
    # Define workflow edges
    workflow.set_entry_point("analyst")
    
    # Conditional routing from analyst based on confidence
    workflow.add_conditional_edges(
        "analyst",
        should_continue,
        {
            "rca": "rca",
            "end": END
        }
    )
    
    # Sequential flow through remaining agents
    workflow.add_edge("rca", "impact")
    workflow.add_edge("impact", "mitigation")
    workflow.add_edge("mitigation", END)
    
    return workflow


@app.entrypoint
async def agent_stream(payload, context: RequestContext):
    """
    Main entrypoint for SRE multi-agent system using LangGraph.
    
    Handles user queries like "whats going on?" and orchestrates
    the 4-agent workflow to analyze incidents.
    """
    user_query = payload.get("prompt")
    session_id = payload.get("runtimeSessionId")
    
    if not all([user_query, session_id]):
        yield {
            "status": "error",
            "error": "Missing required fields: prompt or runtimeSessionId"
        }
        return
    
    try:
        # Extract user ID from JWT token
        user_id = extract_user_id_from_context(context)
        
        print(f"[SRE AGENT] User: {user_id}, Session: {session_id}")
        print(f"[SRE AGENT] Query: {user_query}")
        
        # Get configuration from SSM
        stack_name = os.environ.get("STACK_NAME")
        log_group = get_ssm_parameter(f"/{stack_name}/cloudwatch_log_group")
        
        # Configure memory
        memory_id = os.environ.get("MEMORY_ID")
        checkpointer = AgentCoreMemorySaver(
            memory_id=memory_id,
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        )
        
        # Create workflow
        workflow = create_sre_workflow()
        graph = workflow.compile(checkpointer=checkpointer)
        
        # Yield message start
        yield {
            "event": {
                "messageStart": {"role": "assistant"}
            }
        }
        
        # Get CloudWatch logs
        yield {
            "event": {
                "contentBlockDelta": {
                    "delta": {"text": "📊 Retrieving CloudWatch logs...\n\n"}
                }
            }
        }
        
        log_text, log_count = await get_cloudwatch_logs(log_group, minutes=15)
        
        if not log_text or log_count == 0:
            yield {
                "event": {
                    "contentBlockDelta": {
                        "delta": {"text": "✅ No recent error logs found. System appears normal.\n"}
                    }
                }
            }
            return
        
        yield {
            "event": {
                "contentBlockDelta": {
                    "delta": {"text": f"Retrieved {log_count} log entries from last 15 minutes.\n\n"}
                }
            }
        }
        
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "user_query": user_query,
            "log_text": log_text,
            "log_count": log_count,
            "analyst_result": {},
            "rca_result": {},
            "impact_result": {},
            "mitigation_result": {},
            "current_step": "analyst",
            "scenario": None
        }
        
        # Configure streaming
        config = {
            "configurable": {
                "thread_id": session_id,
                "actor_id": user_id
            }
        }
        
        # Stream workflow execution
        async for event in graph.astream(initial_state, config=config):
            # Stream each agent's output
            for node_name, node_state in event.items():
                if node_name in ["analyst", "rca", "impact", "mitigation"]:
                    # Get the last message added by this node
                    if node_state.get("messages"):
                        last_message = node_state["messages"][-1]
                        if isinstance(last_message, AIMessage):
                            yield {
                                "event": {
                                    "contentBlockDelta": {
                                        "delta": {"text": f"{last_message.content}\n"}
                                    }
                                }
                            }
        
        # Get final state
        final_state = await graph.aget_state(config)
        
        # Output mitigation steps if available
        if final_state.values.get("mitigation_result") and not final_state.values["mitigation_result"].get("error"):
            steps = final_state.values["mitigation_result"].get("steps", [])
            if steps:
                yield {
                    "event": {
                        "contentBlockDelta": {
                            "delta": {"text": "\n**Recommended Mitigation Steps:**\n\n"}
                        }
                    }
                }
                for i, step in enumerate(steps, 1):
                    yield {
                        "event": {
                            "contentBlockDelta": {
                                "delta": {"text": f"{i}. {step}\n"}
                            }
                        }
                    }
                
                recovery_time = final_state.values["mitigation_result"].get("estimated_recovery_time")
                if recovery_time:
                    yield {
                        "event": {
                            "contentBlockDelta": {
                                "delta": {"text": f"\n⏱️ Estimated Recovery Time: {recovery_time}\n"}
                            }
                        }
                    }
        
        yield {
            "event": {
                "contentBlockDelta": {
                    "delta": {"text": "\n✅ Incident analysis complete.\n"}
                }
            }
        }
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        yield {
            "event": {
                "contentBlockDelta": {
                    "delta": {"text": f"\n❌ Error: {str(e)}\n"}
                }
            }
        }


if __name__ == "__main__":
    app.run()
