from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
import time
from datetime import datetime
from llm_providers import get_llm_provider
from document_parsers import parse_document
from models import WorkflowStep

class DocumentState(TypedDict):
    filename: str
    raw_content: bytes
    content_type: str
    parsed_text: str
    document_type: str
    generated_code: str
    language: str
    workflow_steps: List[WorkflowStep]
    start_time: float

class DocumentWorkflow:
    def __init__(self):
        self.llm = get_llm_provider()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for document processing"""
        workflow = StateGraph(DocumentState)
        
        # Add nodes
        workflow.add_node("parse_document", self.parse_document_node)
        workflow.add_node("analyze_content", self.analyze_content_node)
        workflow.add_node("generate_code", self.generate_code_node)
        workflow.add_node("validate_output", self.validate_output_node)
        
        # Define edges
        workflow.add_edge(START, "parse_document")
        workflow.add_edge("parse_document", "analyze_content")
        workflow.add_edge("analyze_content", "generate_code")
        workflow.add_edge("generate_code", "validate_output")
        workflow.add_edge("validate_output", END)
        
        return workflow.compile()
    
    async def parse_document_node(self, state: DocumentState) -> DocumentState:
        """Parse uploaded document and extract text content"""
        step_start = time.time()
        
        try:
            parsed_result = parse_document(
                content=state["raw_content"],
                filename=state["filename"],
                content_type=state["content_type"]
            )
            
            state["parsed_text"] = parsed_result["text"]
            state["document_type"] = parsed_result["document_type"]
            
            step = WorkflowStep(
                step_name="parse_document",
                status="completed",
                duration=time.time() - step_start,
                output=f"Extracted {len(state['parsed_text'])} characters"
            )
            
        except Exception as e:
            step = WorkflowStep(
                step_name="parse_document",
                status="failed",
                duration=time.time() - step_start,
                output=f"Error: {str(e)}"
            )
            raise
        
        state["workflow_steps"].append(step)
        return state
    
    async def analyze_content_node(self, state: DocumentState) -> DocumentState:
        """Analyze document content to determine code generation approach"""
        step_start = time.time()
        
        analysis_prompt = f"""
        Analyze this documentation and determine:
        1. What type of code should be generated (API client, CLI tool, web app, etc.)
        2. What programming language would be most appropriate
        3. Key components and features to implement
        
        Document content:
        {state["parsed_text"][:2000]}...
        
        Respond with a brief analysis focusing on the code generation strategy.
        """
        
        try:
            analysis = await self.llm.generate(analysis_prompt)
            
            # Extract language from analysis (simple heuristic)
            language = self._detect_language(analysis, state["parsed_text"])
            state["language"] = language
            
            step = WorkflowStep(
                step_name="analyze_content",
                status="completed",
                duration=time.time() - step_start,
                output=f"Detected language: {language}"
            )
            
        except Exception as e:
            step = WorkflowStep(
                step_name="analyze_content",
                status="failed",
                duration=time.time() - step_start,
                output=f"Error: {str(e)}"
            )
            raise
        
        state["workflow_steps"].append(step)
        return state
    
    async def generate_code_node(self, state: DocumentState) -> DocumentState:
        """Generate code based on the analyzed documentation"""
        step_start = time.time()
        
        code_prompt = f"""
        Based on this API documentation, generate complete, production-ready {state["language"]} code.
        
        Requirements:
        - Include all necessary imports and dependencies
        - Add proper error handling
        - Include example usage
        - Add comments explaining key functionality
        - Make it ready to run
        
        Documentation:
        {state["parsed_text"]}
        
        Generate clean, working {state["language"]} code:
        """
        
        try:
            generated_code = await self.llm.generate(code_prompt)
            state["generated_code"] = self._clean_code_output(generated_code)
            
            step = WorkflowStep(
                step_name="generate_code",
                status="completed",
                duration=time.time() - step_start,
                output=f"Generated {len(state['generated_code'])} characters of code"
            )
            
        except Exception as e:
            step = WorkflowStep(
                step_name="generate_code",
                status="failed",
                duration=time.time() - step_start,
                output=f"Error: {str(e)}"
            )
            raise
        
        state["workflow_steps"].append(step)
        return state
    
    async def validate_output_node(self, state: DocumentState) -> DocumentState:
        """Validate and clean up the generated code"""
        step_start = time.time()
        
        # Basic validation - check if code looks reasonable
        code = state["generated_code"]
        validation_issues = []
        
        if len(code) < 50:
            validation_issues.append("Code too short")
        
        if state["language"].lower() == "python" and "import" not in code:
            validation_issues.append("Missing import statements")
        
        status = "completed" if not validation_issues else "completed_with_warnings"
        output = "Code validated successfully" if not validation_issues else f"Warnings: {', '.join(validation_issues)}"
        
        step = WorkflowStep(
            step_name="validate_output",
            status=status,
            duration=time.time() - step_start,
            output=output
        )
        
        state["workflow_steps"].append(step)
        return state
    
    def _detect_language(self, analysis: str, content: str) -> str:
        """Simple language detection based on content"""
        analysis_lower = analysis.lower()
        content_lower = content.lower()
        
        if "python" in analysis_lower or "flask" in content_lower or "django" in content_lower:
            return "Python"
        elif "javascript" in analysis_lower or "node" in content_lower or "npm" in content_lower:
            return "JavaScript"
        elif "java" in analysis_lower and "javascript" not in analysis_lower:
            return "Java"
        elif "curl" in content_lower or "api" in content_lower:
            return "Python"  # Default to Python for API clients
        else:
            return "Python"  # Default fallback
    
    def _clean_code_output(self, code: str) -> str:
        """Clean up the generated code output"""
        # Remove markdown code blocks if present
        if "```" in code:
            lines = code.split("\n")
            start_idx = 0
            end_idx = len(lines)
            
            for i, line in enumerate(lines):
                if line.strip().startswith("```"):
                    if start_idx == 0:
                        start_idx = i + 1
                    else:
                        end_idx = i
                        break
            
            code = "\n".join(lines[start_idx:end_idx])
        
        return code.strip()
    
    async def run(self, filename: str, content: bytes, content_type: str) -> dict:
        """Execute the complete workflow"""
        initial_state = DocumentState(
            filename=filename,
            raw_content=content,
            content_type=content_type,
            parsed_text="",
            document_type="",
            generated_code="",
            language="",
            workflow_steps=[],
            start_time=time.time()
        )
        
        final_state = await self.workflow.ainvoke(initial_state)
        
        return {
            "generated_code": final_state["generated_code"],
            "language": final_state["language"],
            "processing_time": time.time() - final_state["start_time"],
            "workflow_steps": final_state["workflow_steps"]
        }