from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ProcessingRequest(BaseModel):
    filename: str
    content_type: str

class WorkflowStep(BaseModel):
    step_name: str
    status: str
    duration: float
    output: Optional[str] = None

class ProcessingResponse(BaseModel):
    success: bool
    filename: str
    document_intelligence: dict
    processing_time: float
    workflow_steps: List[WorkflowStep]
    timestamp: datetime = datetime.now()

class DocumentMetadata(BaseModel):
    filename: str
    file_size: int
    content_type: str
    upload_time: datetime
    processing_status: str

class LLMProvider(BaseModel):
    name: str
    type: str  # "bedrock" or "sagemaker"
    model_name: str
    endpoint_url: Optional[str] = None