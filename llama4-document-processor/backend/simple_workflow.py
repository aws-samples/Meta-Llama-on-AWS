import time
from llm_providers import get_llm_provider
from document_parsers import parse_document
from models import WorkflowStep

class SimpleDocumentProcessor:
    def __init__(self):
        self.llm = get_llm_provider()
    
    async def process_document(self, filename: str, content: bytes, content_type: str) -> dict:
        """Simple document processing without LangGraph"""
        start_time = time.time()
        workflow_steps = []
        
        # Step 1: Parse document
        step_start = time.time()
        try:
            parsed_result = parse_document(content, filename, content_type)
            parsed_text = parsed_result["text"]
            
            workflow_steps.append(WorkflowStep(
                step_name="parse_document",
                status="completed",
                duration=time.time() - step_start,
                output=f"Extracted {len(parsed_text)} characters"
            ))
        except Exception as e:
            workflow_steps.append(WorkflowStep(
                step_name="parse_document", 
                status="failed",
                duration=time.time() - step_start,
                output=f"Error: {str(e)}"
            ))
            raise
        
        # Step 2: Generate code
        step_start = time.time()
        try:
            code_prompt = f"""
            Based on this API documentation, generate complete, production-ready Python code.
            
            Requirements:
            - Include all necessary imports and dependencies
            - Add proper error handling
            - Include example usage
            - Add comments explaining key functionality
            - Make it ready to run
            
            Documentation:
            {parsed_text}
            
            Generate clean, working Python code:
            """
            
            generated_code = await self.llm.generate(code_prompt)
            
            # Clean up code output
            if "```" in generated_code:
                lines = generated_code.split("\n")
                start_idx = 0
                end_idx = len(lines)
                
                for i, line in enumerate(lines):
                    if line.strip().startswith("```"):
                        if start_idx == 0:
                            start_idx = i + 1
                        else:
                            end_idx = i
                            break
                
                generated_code = "\n".join(lines[start_idx:end_idx])
            
            workflow_steps.append(WorkflowStep(
                step_name="generate_code",
                status="completed", 
                duration=time.time() - step_start,
                output=f"Generated {len(generated_code)} characters of code"
            ))
            
        except Exception as e:
            workflow_steps.append(WorkflowStep(
                step_name="generate_code",
                status="failed",
                duration=time.time() - step_start, 
                output=f"Error: {str(e)}"
            ))
            raise
        
        return {
            "generated_code": generated_code.strip(),
            "language": "Python",
            "processing_time": time.time() - start_time,
            "workflow_steps": workflow_steps
        }