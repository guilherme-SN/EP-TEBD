from typing import Optional
from pydantic import BaseModel

# Modelo para resposta da API
class EmbeddingResponse(BaseModel):
    """Resposta após criar/consultar embedding"""
    id: int                             # ID gerado pelo Milvus
    label: str                          # Label correspondente
    status: str                         # "created", "found", etc.
    distance: Optional[float] = None    # Para buscas por similaridade
