from typing import Dict, List
from starlette.concurrency import run_in_threadpool
from simple_workflow import compiled_graph, WorkflowState  # Use the compiled graph!
from llm_providers import get_llm_provider

class DocumentProcessor:
    def __init__(self):
        self.graph = compiled_graph
        self.llm_provider = get_llm_provider()

    async def process_document(self, filename: str, content: bytes, content_type: str) -> Dict:
        initial_state: WorkflowState = {
            "filename": filename,
            "content": content,
            "content_type": content_type,
            "parsed_text": "",
            "document_intelligence": {},
            "workflow_log": [],
            "processing_time": 0.0
        }
        # Run blocking synchronous code safely in async context:
        final_state = await compiled_graph.ainvoke(initial_state)
        
        return {
            "document_intelligence": final_state.get("document_intelligence", {}),
            "processing_time": final_state.get("processing_time", 0.0),
            "workflow_steps": final_state.get("workflow_log", [])
        }
        
    async def get_available_models(self) -> List[str]:
        return await self.llm_provider.get_available_models()
