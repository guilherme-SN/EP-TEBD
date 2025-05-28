
from fastapi import APIRouter, Depends
from services.embedding_service import EmbeddingService
from schemas.requests.embedding_create import EmbeddingCreate
from schemas.requests.embedding_from_image_create import EmbeddingFromImageCreate
from schemas.responses.embedding_response import EmbeddingResponse

router = APIRouter()

@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embedding(
    data: EmbeddingCreate,
    embedding_service: EmbeddingService = Depends(EmbeddingService)
):
    """Endpoint para criação de embeddings"""
    return embedding_service.create_embedding(data)

@router.post("/embeddings-from-image", response_model=EmbeddingResponse)
async def create_embedding_from_image(
        data: EmbeddingFromImageCreate,
        embedding_service: EmbeddingService = Depends(EmbeddingService)
):
    """Endpoint para criação de embeddings"""
    return embedding_service.create_embedding_from_image(data)

