from typing import List
from schemas.responses.embedding_response import EmbeddingResponse
from pydantic import BaseModel

class EmbeddingSearchResponse(BaseModel):
    """Schema para a resposta de uma busca por similaridade."""
    similar_embeddings: List[EmbeddingResponse]