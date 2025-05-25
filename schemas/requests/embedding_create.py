from typing import List, Optional
from pydantic import BaseModel

# Modelo para criação de embeddings
class EmbeddingCreate(BaseModel):
    """Schema para criação de embeddings"""
    embedding: List[float]                  # Lista de 128 floats (por exemplo)
    label: str                              # Ex: "letra_A"
    source: Optional[str] = "user_upload"   # Origem do dado