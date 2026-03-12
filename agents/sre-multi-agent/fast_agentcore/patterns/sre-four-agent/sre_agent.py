# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SRE Four-Agent System integrated with AgentCore Runtime.

This module wraps the 4-agent incident response orchestrator (Analyst, RCA, Impact, Mitigation)
to work with AgentCore's runtime model and streaming interface.

Version: 1.0.1 - Fixed auth import
"""

import os
import sys
import traceback
from datetime import datetime, timezone
from typing import AsyncIterator, Dict, Any

from bedrock_agentcore.runtime import BedrockAgentCoreApp, RequestContext
from langgraph_checkpoint_aws import AgentCoreMemorySaver

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from orchestration.four_agent.orchestrator import PhaseTwoOrchestrator
from orchestration.four_agent.analyst_agent import AnalystAgent
from orchestration.four_agent.rca_agent import RCAAgent
from orchestration.four_agent.impact_agent import ImpactAgent
from orchestration.four_agent.mitigation_agent import MitigationCommsAgent
from orchestration.four_agent.scenario_loader import ScenarioSnapshot, ScenarioMetadata, TimeWindow
from orchestration.four_agent.schema import Severity
from orchestration.four_agent.bedrock_kb_reader import get_kb_retrievals, clear_kb_retrievals


def extract_user_id_from_context(context: RequestContext) -> str:
    """
    Extract user ID from request context.
    
    Args:
        context: AgentCore request context
        
    Returns:
        User identifier (defaults to 'anonymous' if not found)
    """
    try:
        # Try to get user ID from JWT claims
        if hasattr(context, 'identity') and context.identity:
            return context.identity.get('sub', 'anonymous')
        return 'anonymous'
    except Exception:
        return 'anonymous'

app = BedrockAgentCoreApp()


async def create_orchestrator(user_id: str, session_id: str) -> PhaseTwoOrchestrator:
    """
    Create the 4-agent orchestrator with all agents initialized.
    
    Args:
        user_id: User identifier from JWT token
        session_id: Session identifier for conversation memory
        
    Returns:
        Configured PhaseTwoOrchestrator instance
    """
    print(f"[SRE] Creating 4-agent orchestrator for user: {user_id}, session: {session_id}")
    
    # Initialize all 4 agents
    analyst = AnalystAgent()
    rca = RCAAgent()
    impact = ImpactAgent()
    mitigation = MitigationCommsAgent()
    
    # Create orchestrator
    orchestrator = PhaseTwoOrchestrator(
        analyst_agent=analyst,
        rca_agent=rca,
        impact_agent=impact,
        mitigation_agent=mitigation,
        demo_mode=False,
        incident_id=session_id
    )
    
    print("[SRE] 4-agent orchestrator created successfully")
    return orchestrator


def create_scenario_from_query(query: str, session_id: str) -> tuple[ScenarioSnapshot, Dict[str, Any]]:
    """
    Convert a user query into a ScenarioSnapshot for the orchestrator.
    
    Args:
        query: User's query/prompt
        session_id: Session identifier
        
    Returns:
        Tuple of (ScenarioSnapshot, metadata dict with details for user display)
    """
    import boto3
    import json
    from datetime import timedelta
    from collections import Counter
    
    now = datetime.now(timezone.utc)
    monitoring_data = {
        "query": query,
        "timestamp": now.isoformat()
    }
    
    # Metadata for user display
    display_metadata = {
        "cloudwatch_fetched": False,
        "log_count": 0,
        "time_range": None,
        "services_found": [],
        "error_count": 0,
        "log_stream": None
    }
    
    # Check if user is asking about CloudWatch logs
    if "/aws/banking/system-logs" in query.lower() or "cloudwatch" in query.lower():
        try:
            logs_client = boto3.client('logs')
            
            # Get recent logs from the last hour
            start_time = now - timedelta(hours=1)
            
            print(f"[SRE] Fetching CloudWatch logs from /aws/banking/system-logs")
            
            response = logs_client.filter_log_events(
                logGroupName='/aws/banking/system-logs',
                startTime=int(start_time.timestamp() * 1000),
                endTime=int(now.timestamp() * 1000),
                limit=500  # Get last 500 events
            )
            
            if response.get('events'):
                # Parse log events
                log_events = []
                services = []
                error_count = 0
                
                for event in response['events']:
                    try:
                        log_data = json.loads(event['message'])
                        log_events.append(log_data)
                        
                        # Track services
                        if 'service' in log_data:
                            services.append(log_data['service'])
                        
                        # Count errors
                        if log_data.get('level') == 'ERROR' or log_data.get('http', {}).get('status_code', 0) >= 400:
                            error_count += 1
                    except:
                        pass
                
                # Get log stream name
                log_stream = response['events'][0].get('logStreamName', 'unknown') if response['events'] else None
                
                monitoring_data['cloudwatch_logs'] = {
                    'log_group': '/aws/banking/system-logs',
                    'event_count': len(log_events),
                    'time_range': {
                        'start': start_time.isoformat(),
                        'end': now.isoformat()
                    },
                    'events': log_events[:100]  # Include first 100 events
                }
                
                # Update display metadata
                display_metadata['cloudwatch_fetched'] = True
                display_metadata['log_count'] = len(log_events)
                display_metadata['time_range'] = f"{start_time.strftime('%H:%M')} - {now.strftime('%H:%M UTC')}"
                display_metadata['services_found'] = list(set(services))
                display_metadata['error_count'] = error_count
                display_metadata['log_stream'] = log_stream
                
                print(f"[SRE] Fetched {len(log_events)} log events from CloudWatch")
                print(f"[SRE] Services: {', '.join(set(services))}")
                print(f"[SRE] Errors found: {error_count}")
            else:
                monitoring_data['cloudwatch_logs'] = {
                    'error': 'No logs found in the specified time range'
                }
                print("[SRE] No logs found in CloudWatch")
                
        except Exception as e:
            print(f"[SRE] Error fetching CloudWatch logs: {e}")
            monitoring_data['cloudwatch_logs'] = {
                'error': str(e)
            }
    
    scenario = ScenarioSnapshot(
        metadata=ScenarioMetadata(
            key=f"query_{session_id[:8]}",
            description=query,
            severity=Severity.SEV_3  # Default to SEV-3 for user queries
        ),
        window=TimeWindow(
            start=now,
            end=now
        ),
        monitoring=monitoring_data,
        additional_sources={},
        _incident_id=session_id  # Use the private field name
    )
    
    return scenario, display_metadata


async def stream_orchestrator_results(
    orchestrator: PhaseTwoOrchestrator, 
    scenario: ScenarioSnapshot,
    display_metadata: Dict[str, Any],
    session_id: str,
    user_id: str
) -> AsyncIterator[Dict[str, Any]]:
    """
    Stream results from the 4-agent orchestrator in real-time as each agent completes.
    
    Uses LangGraph's streaming API to get updates as each node (agent) completes.
    
    Args:
        orchestrator: The orchestrator instance
        scenario: The scenario to analyze
        display_metadata: Metadata about data sources for user display
        session_id: Session ID for memory storage
        user_id: User ID for memory storage
    
    Yields:
        Dict chunks representing agent progress and responses as they complete
    """
    try:
        # Yield initial status with data source information
        status_parts = ["🔍 Starting 4-agent incident response workflow...\n"]
        
        if display_metadata.get('cloudwatch_fetched'):
            status_parts.append(f"\n📊 **Data Sources:**")
            status_parts.append(f"\n- CloudWatch Logs: `/aws/banking/system-logs`")
            status_parts.append(f"\n- Log Stream: `{display_metadata.get('log_stream', 'N/A')}`")
            status_parts.append(f"\n- Time Range: {display_metadata.get('time_range', 'N/A')}")
            status_parts.append(f"\n- Total Events: {display_metadata.get('log_count', 0):,}")
            status_parts.append(f"\n- Services: {', '.join(display_metadata.get('services_found', []))}")
            status_parts.append(f"\n- Errors Found: {display_metadata.get('error_count', 0)}")
            status_parts.append(f"\n\n- Knowledge Base: `YOUR_KB_ID` (SRE policies & runbooks)")
        
        status_parts.append("\n\n⚙️ Initializing agents: Analyst → RCA → Impact → Mitigation\n")
        
        yield {
            "type": "status",
            "content": "".join(status_parts)
        }
        
        # Clear KB tracking before running
        clear_kb_retrievals()
        
        # Run agents sequentially and stream results as each completes
        # Initialize runtime state
        runtime = orchestrator._initial_runtime(scenario)
        
        # Agent execution order and display info
        agent_nodes = ["analyst", "rca", "impact", "mitigation"]
        agent_display = {
            "analyst": ("Analyst", "🔎 Analyzing logs and identifying anomalies..."),
            "rca": ("RCA", "🔬 Performing root cause analysis..."),
            "impact": ("Impact", "📈 Assessing business impact..."),
            "mitigation": ("Mitigation", "🛠️ Generating mitigation recommendations...")
        }
        
        # Execute each agent node sequentially and stream results
        for idx, node_name in enumerate(agent_nodes, 1):
            # Get the agent node function
            if node_name == "analyst":
                await orchestrator._analyst_node(runtime)
            elif node_name == "rca":
                await orchestrator._rca_node(runtime)
            elif node_name == "impact":
                await orchestrator._impact_node(runtime)
            elif node_name == "mitigation":
                await orchestrator._mitigation_node(runtime)
            
            # Get the latest response from this agent
            if runtime.result.responses and len(runtime.result.responses) >= idx:
                response = runtime.result.responses[idx - 1]
                agent_name, agent_desc = agent_display[node_name]
                
                # Add agent-specific context
                agent_intro = f"\n\n{'='*60}\n"
                agent_intro += f"**Agent {idx}/4: {agent_name}**\n"
                agent_intro += f"{'='*60}\n\n"
                agent_intro += f"{agent_desc}\n\n"
                
                # Combine intro with agent response
                full_content = agent_intro + response.payload.summary
                
                # Add details if available - format as pretty JSON
                if response.payload.details:
                    import json
                    try:
                        # Convert details to pretty-printed JSON
                        details_json = json.dumps(response.payload.details, indent=2, ensure_ascii=False)
                        full_content += f"\n\n**Additional Details:**\n```json\n{details_json}\n```"
                    except (TypeError, ValueError):
                        # Fallback to string representation if JSON serialization fails
                        full_content += f"\n\n**Additional Details:**\n{response.payload.details}"
                
                # Yield this agent's response immediately
                yield {
                    "type": "agent_response",
                    "agent": agent_name,
                    "content": full_content,
                    "details": response.payload.details
                }
        
        # Run summary node to complete the workflow
        await orchestrator._summary_node(runtime)
        
        # Get the final result
        result = runtime.result
        
        # Get KB retrievals that occurred during execution
        kb_retrievals = get_kb_retrievals()
        
        # Display KB retrieval summary if any occurred
        if kb_retrievals:
            kb_summary = ["\n\n📚 **Knowledge Base Queries:**"]
            kb_summary.append(f"\nTotal Retrievals: {len(kb_retrievals)}")
            
            # Group by unique sources
            all_sources = set()
            for retrieval in kb_retrievals:
                all_sources.update(retrieval['sources'])
            
            if all_sources:
                kb_summary.append(f"\n\n**Policy Documents Retrieved:**")
                for source in sorted(all_sources):
                    kb_summary.append(f"\n- {source}")
            
            # Show sample queries
            kb_summary.append(f"\n\n**Sample Queries:**")
            for i, retrieval in enumerate(kb_retrievals[:3], 1):
                query_preview = retrieval['query'][:60] + "..." if len(retrieval['query']) > 60 else retrieval['query']
                kb_summary.append(f"\n{i}. \"{query_preview}\" ({retrieval['result_count']} results)")
            
            if len(kb_retrievals) > 3:
                kb_summary.append(f"\n... and {len(kb_retrievals) - 3} more queries")
            
            yield {
                "type": "kb_summary",
                "content": "".join(kb_summary)
            }
        
        # Stream the final summary
        if result.summary:
            summary_header = "\n\n" + "="*60 + "\n"
            summary_header += "**📋 Incident Analysis Complete**\n"
            summary_header += "="*60 + "\n\n"
            
            yield {
                "type": "summary",
                "content": summary_header + result.summary.to_markdown()
            }
        
        yield {
            "type": "complete",
            "content": "\n\n✅ 4-agent workflow completed successfully"
        }
        
        # Save analysis results to AgentCore memory
        memory_id = os.environ.get("MEMORY_ID")
        if memory_id:
            try:
                from langchain_core.messages import HumanMessage, AIMessage
                
                checkpointer = AgentCoreMemorySaver(
                    memory_id=memory_id,
                    region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
                )
                
                config = {"configurable": {"thread_id": session_id, "user_id": user_id, "actor_id": user_id}}
                
                # Get current checkpoint
                checkpoint_tuple = await checkpointer.aget(config)
                messages = []
                checkpoint_id = None
                
                # Handle both tuple and dict responses from aget()
                if checkpoint_tuple:
                    if hasattr(checkpoint_tuple, 'checkpoint'):
                        # It's a CheckpointTuple
                        checkpoint = checkpoint_tuple.checkpoint
                    elif isinstance(checkpoint_tuple, dict):
                        # It's a dict (legacy format)
                        checkpoint = checkpoint_tuple
                    else:
                        checkpoint = None
                        
                    if checkpoint:
                        messages = checkpoint.get("channel_values", {}).get("messages", [])
                        checkpoint_id = checkpoint.get("id")
                
                # Build comprehensive analysis summary for memory
                analysis_summary = f"Incident Analysis Results:\n\n"
                
                if display_metadata.get('cloudwatch_fetched'):
                    analysis_summary += f"Data Sources:\n"
                    analysis_summary += f"- CloudWatch Logs: {display_metadata.get('log_count', 0)} events\n"
                    analysis_summary += f"- Services: {', '.join(display_metadata.get('services_found', []))}\n"
                    analysis_summary += f"- Errors: {display_metadata.get('error_count', 0)}\n\n"
                
                # Add agent responses
                for idx, response in enumerate(result.responses, 1):
                    agent_name = response.sender.value
                    analysis_summary += f"{agent_name} Agent:\n{response.payload.summary}\n\n"
                
                # Add final summary
                if result.summary:
                    analysis_summary += f"Final Summary:\n{result.summary.to_markdown()}\n"
                
                # Add KB retrieval info
                if kb_retrievals:
                    all_sources = set()
                    for retrieval in kb_retrievals:
                        all_sources.update(retrieval['sources'])
                    if all_sources:
                        analysis_summary += f"\nKnowledge Base Documents Used:\n"
                        for source in sorted(all_sources):
                            analysis_summary += f"- {source}\n"
                
                # Add messages to memory
                messages.append(HumanMessage(content=scenario.metadata.description))
                messages.append(AIMessage(content=analysis_summary))
                
                # Create checkpoint with ID
                import uuid
                new_checkpoint_id = checkpoint_id or str(uuid.uuid4())
                
                # Save to memory - aput requires: config, checkpoint, metadata, new_versions
                await checkpointer.aput(
                    config,
                    {
                        "id": new_checkpoint_id,
                        "channel_values": {"messages": messages},
                        "channel_versions": {"messages": len(messages)}
                    },
                    {"source": "update", "step": len(messages), "writes": None},
                    {"messages": len(messages)}  # new_versions parameter (4th argument)
                )
                print(f"[SRE] Saved analysis results to AgentCore memory")
            except Exception as e:
                print(f"[SRE] Error saving to memory: {e}")
                traceback.print_exc()
                traceback.print_exc()
        
    except Exception as e:
        yield {
            "type": "error",
            "content": f"Error in orchestrator: {str(e)}"
        }
        traceback.print_exc()


@app.entrypoint
async def agent_stream(payload, context: RequestContext):
    """
    Main entrypoint for the SRE 4-agent system.
    
    This function receives requests from AgentCore Runtime, extracts the user query,
    and either runs the full 4-agent workflow or provides conversational responses
    about previous analysis.
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
        # Extract user ID securely from JWT token
        user_id = extract_user_id_from_context(context)
        
        print(f"[SRE] Received query for user: {user_id}, session: {session_id}")
        print(f"[SRE] Query: {user_query}")
        
        # Detect if this is a request for incident analysis or a follow-up question
        is_incident_analysis = _is_incident_analysis_request(user_query)
        
        if is_incident_analysis:
            print("[SRE] Detected incident analysis request - running 4-agent workflow")
            
            # Create orchestrator
            orchestrator = await create_orchestrator(user_id, session_id)
            
            # Create scenario from query and get display metadata
            scenario, display_metadata = create_scenario_from_query(user_query, session_id)
            
            # Stream results with metadata
            async for chunk in stream_orchestrator_results(orchestrator, scenario, display_metadata, session_id, user_id):
                # Convert to LangGraph-compatible format for frontend
                if chunk["type"] == "agent_response":
                    yield {
                        "type": "AIMessageChunk",
                        "content": [{
                            "type": "text",
                            "text": chunk['content']
                        }]
                    }
                elif chunk["type"] in ("status", "summary", "complete", "kb_summary"):
                    yield {
                        "type": "AIMessageChunk",
                        "content": [{
                            "type": "text",
                            "text": chunk["content"]
                        }]
                    }
                elif chunk["type"] == "error":
                    yield {
                        "type": "AIMessageChunk",
                        "content": [{
                            "type": "text",
                            "text": f"❌ {chunk['content']}"
                        }]
                    }
            
            print("[SRE] 4-agent workflow completed successfully")
        else:
            print("[SRE] Detected follow-up question - providing conversational response")
            
            # Provide a conversational response about the system
            async for chunk in _handle_conversational_query(user_query, session_id, user_id):
                yield chunk
        
        # Send completion marker
        yield {
            "type": "AIMessageChunk",
            "response_metadata": {"stop_reason": "end_turn"}
        }
            
    except Exception as e:
        error_msg = str(e) if str(e) else f"{type(e).__name__}: {repr(e)}"
        print(f"[SRE ERROR] Error in agent_stream: {error_msg}")
        traceback.print_exc()
        yield {
            "type": "AIMessageChunk",
            "content": [{
                "type": "text",
                "text": f"Error: {error_msg}"
            }],
            "response_metadata": {"stop_reason": "error"}
        }


def _is_incident_analysis_request(query: str) -> bool:
    """
    Determine if the query is requesting a new incident analysis.
    
    Args:
        query: User's query text
        
    Returns:
        True if this should trigger the 4-agent workflow, False for conversational responses
    """
    query_lower = query.lower()
    
    # Keywords that indicate incident analysis request
    analysis_keywords = [
        "analyze",
        "analyze the logs",
        "check for incidents",
        "investigate",
        "find incidents",
        "detect incidents",
        "scan logs",
        "review logs",
        "/aws/banking/system-logs"  # Specific log group reference
    ]
    
    # Check if query contains analysis keywords
    for keyword in analysis_keywords:
        if keyword in query_lower:
            return True
    
    return False


async def _handle_conversational_query(query: str, session_id: str, user_id: str) -> AsyncIterator[Dict[str, Any]]:
    """
    Handle conversational queries about the SRE system using AgentCore memory.
    
    Args:
        query: User's conversational query
        session_id: Session identifier for memory access
        user_id: User identifier for memory access
        
    Yields:
        Response chunks in LangGraph format
    """
    import boto3
    import json
    
    # Get memory ID from environment
    memory_id = os.environ.get("MEMORY_ID")
    if not memory_id:
        print("[SRE] Warning: MEMORY_ID not set, conversational mode will have no context")
    
    # Initialize AgentCore memory
    checkpointer = None
    conversation_history = []
    
    if memory_id:
        try:
            checkpointer = AgentCoreMemorySaver(
                memory_id=memory_id,
                region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
            )
            
            # Retrieve conversation history from memory
            config = {"configurable": {"thread_id": session_id, "user_id": user_id, "actor_id": user_id}}
            checkpoint_tuple = await checkpointer.aget(config)
            
            # Handle both tuple and dict responses from aget()
            if checkpoint_tuple:
                if hasattr(checkpoint_tuple, 'checkpoint'):
                    # It's a CheckpointTuple
                    checkpoint = checkpoint_tuple.checkpoint
                elif isinstance(checkpoint_tuple, dict):
                    # It's a dict (legacy format)
                    checkpoint = checkpoint_tuple
                else:
                    checkpoint = None
                    
                if checkpoint:
                    messages = checkpoint.get("channel_values", {}).get("messages", [])
                    # Get last 10 messages for context
                    conversation_history = messages[-10:] if len(messages) > 10 else messages
                    print(f"[SRE] Retrieved {len(conversation_history)} messages from memory")
                else:
                    print("[SRE] No checkpoint data found")
            else:
                print("[SRE] No checkpoint found in memory")
        except Exception as e:
            print(f"[SRE] Error accessing memory: {e}")
            traceback.print_exc()
    
    # Build context from conversation history
    context_messages = []
    for msg in conversation_history:
        if hasattr(msg, 'type'):
            if msg.type == 'human':
                context_messages.append(f"User: {msg.content}")
            elif msg.type == 'ai':
                # Truncate long AI messages
                content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                context_messages.append(f"Assistant: {content}")
    
    # Build context string with better handling
    if context_messages:
        context_str = "\n".join(context_messages[-6:])
        print(f"[SRE] Using {len(context_messages)} messages for context")
    else:
        context_str = "No previous conversation in this session."
        print("[SRE] No conversation history found in memory")
    
    # Use Bedrock with Llama 3.3 70B (same model as the 4-agent system)
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-west-2')
    
    system_prompt = f"""You are an SRE assistant helping users understand the 4-agent incident response system.

The system consists of:
- **Analyst Agent**: Analyzes logs and identifies anomalies
- **RCA Agent**: Performs root cause analysis
- **Impact Agent**: Assesses business impact (TPS, revenue, approvals)
- **Mitigation Agent**: Generates mitigation recommendations

The system analyzes CloudWatch logs from `/aws/banking/system-logs` and retrieves relevant policies from a Knowledge Base (ID: YOUR_KB_ID) containing:
- Business impact baselines (POL-SRE-002)
- Known failure patterns (POL-SRE-001)
- Troubleshooting runbooks (POL-SRE-003)
- Communication templates (POL-SRE-004)

To run a new incident analysis, click the "Test System" button or ask to "analyze the logs".

Previous conversation context:
{context_str}

Answer questions about the system, its capabilities, previous analysis results, and how to use it. Be helpful and concise. If asked about previous analysis, refer to the conversation history above."""

    # Prepare the request for Llama 3.3 70B
    request_body = {
        "prompt": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
        "max_gen_len": 1000,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    response_text = ""
    
    try:
        # Call Bedrock with Llama 3.3 70B streaming
        response = bedrock_runtime.invoke_model_with_response_stream(
            modelId="us.meta.llama3-3-70b-instruct-v1:0",
            body=json.dumps(request_body)
        )
        
        # Stream the response
        for event in response['body']:
            chunk = json.loads(event['chunk']['bytes'].decode())
            
            if 'generation' in chunk:
                text = chunk['generation']
                if text:
                    response_text += text
                    yield {
                        "type": "AIMessageChunk",
                        "content": [{
                            "type": "text",
                            "text": text
                        }]
                    }
        
        # Save to memory if available
        if checkpointer and response_text:
            try:
                from langchain_core.messages import HumanMessage, AIMessage
                import uuid
                
                config = {"configurable": {"thread_id": session_id, "user_id": user_id, "actor_id": user_id}}
                
                # Get current checkpoint
                checkpoint_tuple = await checkpointer.aget(config)
                messages = []
                checkpoint_id = None
                
                # Handle both tuple and dict responses from aget()
                if checkpoint_tuple:
                    if hasattr(checkpoint_tuple, 'checkpoint'):
                        # It's a CheckpointTuple
                        checkpoint = checkpoint_tuple.checkpoint
                    elif isinstance(checkpoint_tuple, dict):
                        # It's a dict (legacy format)
                        checkpoint = checkpoint_tuple
                    else:
                        checkpoint = None
                        
                    if checkpoint:
                        messages = checkpoint.get("channel_values", {}).get("messages", [])
                        checkpoint_id = checkpoint.get("id")
                
                # Add new messages
                messages.append(HumanMessage(content=query))
                messages.append(AIMessage(content=response_text))
                
                # Create checkpoint with ID
                new_checkpoint_id = checkpoint_id or str(uuid.uuid4())
                
                # Save back to memory - aput requires: config, checkpoint, metadata, new_versions
                await checkpointer.aput(
                    config,
                    {
                        "id": new_checkpoint_id,
                        "channel_values": {"messages": messages},
                        "channel_versions": {"messages": len(messages)}
                    },
                    {"source": "update", "step": len(messages), "writes": None},
                    {"messages": len(messages)}  # new_versions parameter (4th argument)
                )
                print(f"[SRE] Saved conversation to memory")
            except Exception as e:
                print(f"[SRE] Error saving to memory: {e}")
                traceback.print_exc()
                traceback.print_exc()
                
    except Exception as e:
        print(f"[SRE] Error in conversational response: {e}")
        traceback.print_exc()
        yield {
            "type": "AIMessageChunk",
            "content": [{
                "type": "text",
                "text": f"I'm here to help! To run a new incident analysis, click the 'Test System' button. For questions about the system, I can explain how the 4-agent workflow works, what data sources it uses, and how to interpret the results."
            }]
        }


if __name__ == "__main__":
    app.run()
