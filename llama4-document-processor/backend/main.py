from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from models import ProcessingRequest, ProcessingResponse

load_dotenv()

app = FastAPI(
    title="Document Processing API",
    description="Convert documentation to code using Llama4 and LangGraph",
    version="1.0.0"
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://YOUR-EC2-IP:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize document processor
processor = DocumentProcessor()

@app.get("/")
async def root():
    return {"message": "Document Processing API with LangGraph and Llama4"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "environment": os.getenv("ENVIRONMENT", "development")}

@app.post("/process-document", response_model=ProcessingResponse)
async def process_document(file: UploadFile = File(...)):
    """
    Process uploaded document and generate code using Llama4
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.pdf', '.docx', '.txt', '.md')):
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Please upload PDF, DOCX, TXT, or MD files."
            )
        
        # Read file content
        content = await file.read()
        
        # Process document through LangGraph workflow
        result = await processor.process_document(
            filename=file.filename,
            content=content,
            content_type=file.content_type
        )
        
        return ProcessingResponse(
            success=True,
            filename=file.filename,
            generated_code=result["generated_code"],
            language=result["language"],
            processing_time=result["processing_time"],
            workflow_steps=result["workflow_steps"]
        )
        
    except Exception as e:
        import traceback
        print(f"ERROR: {str(e)}")
        print(f"TRACEBACK: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/models")
async def get_available_models():
    """Get list of available LLM models"""
    return await processor.get_available_models()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=os.getenv("API_HOST", "0.0.0.0"), 
        port=int(os.getenv("API_PORT", 8000))
    )