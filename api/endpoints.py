from typing import Optional

from fastapi import APIRouter, Depends, UploadFile, File, Form, Path

from schemas.requests.embedding_create import EmbeddingCreate
from schemas.responses.embedding_response import EmbeddingResponse
from schemas.responses.embedding_search_response import EmbeddingSearchResponse
from services.embedding_service import EmbeddingService

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
        image_file: UploadFile = File(...),
        label: str = Form(...),
        source: Optional[str] = Form("user_upload"),
        embedding_service: EmbeddingService = Depends(EmbeddingService)
):
    """Endpoint para criação de embeddings"""
    return await embedding_service.create_embedding_from_image(
        image_file=image_file,
        label=label,
        source=source
    )

@router.post("/embeddings/search", response_model=EmbeddingSearchResponse)
async def search_embeddings_endpoint(
        image_file: UploadFile = File(...),
        top_k: int = Form(5),
        embedding_service: EmbeddingService = Depends(EmbeddingService)
):
    """Endpoint para buscar os embeddings mais similares no banco"""
    return await embedding_service.search_top_k_most_similar(image_file=image_file, top_k=top_k)

@router.delete("/embeddings/{embedding_id}", response_model=dict)
async def delete_embedding_by_id(
        embedding_id: int = Path(...),
        embedding_service: EmbeddingService = Depends(EmbeddingService)
):
    """Endpoint para deletar um embedding pelo ID"""
    return await embedding_service.delete_by_id(embedding_id=embedding_id)
