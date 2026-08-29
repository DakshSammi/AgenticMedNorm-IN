from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class LineItem(BaseModel):
    line_id: str = Field(..., description="Unique identifier for the line")
    text: str = Field(..., description="Verbatim text of the line")
    box: Optional[List[float]] = Field(None, description="Coordinates [x_min, y_min, x_max, y_max]")

class TranscriptionDoc(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the document")
    page_number: int = Field(1, description="Page number of the prescription")
    lines: List[LineItem] = Field(default_factory=list, description="Verbatim lines extracted")
    backend_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata from VLM/OCR engine")
    processing_time_ms: float = Field(0.0, description="Time taken to process in milliseconds")
