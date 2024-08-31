from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple

class ProjectFinetuneData(BaseModel):
    project_dict: Optional[Dict[str, str]] = None
    language: Optional[str] = None
    libraries: Optional[List[str]] = None
    urls: Optional[List[str]] = None
    documents: Optional[List[Tuple[str, bytes]]] = None

class GenerateData(BaseModel):
    file_path: Optional[str] = None
    prior_context: str
    proceeding_context: Optional[str] = None
    max_decode_length: int = 256