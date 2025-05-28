from typing import List, Optional
from pydantic import BaseModel

# Modelo para criação de embeddings
class EmbeddingFromImageCreate(BaseModel):
    image_path: str
    label: str
    source: Optional[str] = "file"
