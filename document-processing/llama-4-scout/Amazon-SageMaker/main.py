from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from starlette.requests import Request
import os
from document_processor import DocumentProcessor
from models import ProcessingResponse  # Only import what we're using


app = FastAPI(
    title="Intelligent Document Processing API",
    description="Extract insights and structured data from documents using Llama4 Scout and LangGraph",
    version="1.0.0"
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for SageMaker
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Template and static file setup
templates = Jinja2Templates(directory="templates")
#app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize document processor
processor = DocumentProcessor()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {"status": "healthy", "environment": os.getenv("ENVIRONMENT", "development")}

@app.post("/process-document", response_model=ProcessingResponse)
async def process_document(file: UploadFile = File(...)):
    """
    Process uploaded document and extract comprehensive intelligence using Llama4 Scout
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
            document_intelligence=result["document_intelligence"],
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
        port=int(os.getenv("API_PORT", 8001))
    )

