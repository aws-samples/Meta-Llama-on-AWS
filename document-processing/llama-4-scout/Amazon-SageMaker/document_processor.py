from simple_workflow import SimpleDocumentProcessor
from typing import Dict

class DocumentProcessor:
    def __init__(self):
        self.workflow = SimpleDocumentProcessor()
    
    async def process_document(self, filename: str, content: bytes, content_type: str) -> Dict:
        """
        Main entry point for document processing
        """
        return await self.workflow.process_document(filename, content, content_type)
    
    async def get_available_models(self):
        """Get available LLM models"""
        return await self.workflow.llm.get_available_models()