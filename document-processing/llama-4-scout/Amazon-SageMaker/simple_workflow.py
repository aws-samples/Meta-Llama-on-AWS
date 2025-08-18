import asyncio
from langgraph.graph import StateGraph
from langgraph.types import Command
from typing import TypedDict, List, Literal
from document_parsers import parse_document
import json
import time
from llm_providers import get_llm_provider

class WorkflowState(TypedDict):
    filename: str
    content: bytes
    content_type: str
    parsed_text: str
    document_intelligence: dict
    workflow_log: List[dict]
    processing_time: float

def node_parse(state: WorkflowState) -> Command[Literal["analyze"]]:
    start_time = time.time()
    result = parse_document(state["content"], state["filename"], state["content_type"])
    parsed_text = result["text"]
    duration = time.time() - start_time
    state["workflow_log"].append({
        "step_name": "parse_document",
        "status": "completed",
        "duration": duration,
        "output": f"Parsed document text size: {len(parsed_text)} chars"
    })
    state["parsed_text"] = parsed_text
    state["processing_time"] += duration
    return Command(update=state, goto="analyze")

# Helper function: Detect if parsed text is meaningful business text
def is_meaningful_text(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 30 or len(stripped.split()) < 10:  # Too short or too few words
        return False
    alphas = sum(c.isalpha() for c in stripped)
    if alphas / max(len(stripped), 1) < 0.5:  # Less than half alphabetic
        return False
    return True

async def node_analyze(state: WorkflowState) -> Command[Literal["complete"]]:
    # PRE-LLM CHECK: Only analyze if text is meaningful
    if not is_meaningful_text(state["parsed_text"]):
        analysis_result = {
            "summary": "Sorry, the document could not be analyzed meaningfully. Please provide a business report or something text-readable.",
            "document_type": "unknown",
            "key_entities": {},
            "structured_data": {},
            "key_insights": [],
            "action_items": [],
            "qa_pairs": [],
            "risk_opportunity": {"risks": [], "opportunities": []}
        }
        state["document_intelligence"] = analysis_result
        state["workflow_log"].append({
            "step_name": "analyze_document",
            "status": "completed",
            "duration": 0,
            "output": "Document was not analyzable (pre-LLM check)"
        })
        state["processing_time"] += 0
        return Command(update=state, goto="complete")

    llm = get_llm_provider()
    prompt = f"""You are a business document analyst specializing in extracting actionable intelligence from corporate documents. Your role is to analyze documents and provide structured insights that help executives make informed decisions.
Analyze the following business document and extract comprehensive intelligence. Focus on identifying key business metrics, strategic insights, and actionable items.
If you determine that the document content is not analyzable (e.g., scanned images, charts without text, handwritten notes), DO NOT provide a random report with confidence. This is very important to ensure you only provide reports for business documents so think about the document that you are processing.
Example 1: For a valid business report:
Input:
[Sample business report text]
Output:
{{
  "summary": "Executive summary of ...",
  "document_type": "report",
  ...
}}
Example 2: For non-analyzable content (e.g., scanned images):
Input:
[Sample images-only document text]
Output:
{{
  "summary": "[ANALYSIS_UNAVAILABLE]"
}}
However, if this is a corporate document - Provide your analysis as valid JSON using this structure:
{{
  "summary": "Executive summary of document purpose and key findings",
  "document_type": "contract|report|manual|research|meeting|proposal|other",
  "key_entities": {{
    "names": ["person names mentioned"],
    "dates": ["important dates and deadlines"],
    "amounts": ["financial figures and metrics"],
    "locations": ["geographic references"],
    "organizations": ["companies and departments"]
  }},
  "structured_data": {{
    "tables": ["table summaries if present"],
    "specifications": ["technical specs or requirements"],
    "requirements": ["business requirements or criteria"]
  }},
  "key_insights": ["strategic business insights and findings"],
  "action_items": [{{
    "task": "specific actionable task",
    "deadline": "deadline if mentioned or null",
    "priority": "high|medium|low"
  }}],
  "qa_pairs": [{{
    "question": "relevant business question",
    "answer": "answer based on document content"
  }}],
  "risk_opportunity": {{
    "risks": ["potential business risks identified"],
    "opportunities": ["growth opportunities and advantages"]
  }}
}}
    
Document:
{state['parsed_text']}
"""
    start_time = time.time()
    llm_response = await llm.generate(prompt)
    try:
        start_idx = llm_response.find("{")
        end_idx = llm_response.rfind("}") + 1
        json_str = llm_response[start_idx:end_idx]
        analysis_result = json.loads(json_str)
    except Exception:
        analysis_result = {"summary": "Unable to parse LLM response", "document_type": "unknown"}

    if analysis_result.get("summary") == "[ANALYSIS_UNAVAILABLE]":
        analysis_result = {
            "summary": "Sorry, the document could not be analyzed meaningfully.",
            "document_type": "unknown",
            "key_entities": {},
            "structured_data": {},
            "key_insights": [],
            "action_items": [],
            "qa_pairs": [],
            "risk_opportunity": {"risks": [], "opportunities": []}
        }

    def is_invalid_output(res):
        if not res.get("summary") or not res.get("key_insights"):
            return True
        summary = res.get("summary", "").lower()
        error_phrases = ["don't know", "unable to analyze", "no information", "cannot", "error"]
        if any(phrase in summary for phrase in error_phrases):
            return True
        return False

    if is_invalid_output(analysis_result):
        analysis_result = {
            "summary": "Sorry, the document could not be analyzed meaningfully. Please provide a business report, compliance report, or something text-readable.",
            "document_type": "unknown",
            "key_entities": {},
            "structured_data": {},
            "key_insights": [],
            "action_items": [],
            "qa_pairs": [],
            "risk_opportunity": {"risks": [], "opportunities": []}
        }
    state["document_intelligence"] = analysis_result
    state["workflow_log"].append({
        "step_name": "analyze_document",
        "status": "completed",
        "duration": time.time() - start_time,
        "output": "Generated detailed document intelligence or error message"
    })
    state["processing_time"] += time.time() - start_time
    return Command(update=state, goto="complete")

def node_complete(state: WorkflowState) -> WorkflowState:
    state["workflow_log"].append({
        "step_name": "complete",
        "status": "completed",
        "duration": 0.0,
        "output": "Workflow complete"
    })
    return state

graph = StateGraph(WorkflowState)
graph.add_node("parse", node_parse)
graph.add_node("analyze", node_analyze)
graph.add_node("complete", node_complete)
graph.add_edge("parse", "analyze")
graph.add_edge("analyze", "complete")
graph.set_entry_point("parse")
compiled_graph = graph.compile()