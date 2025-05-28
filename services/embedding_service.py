import logging
from milvus.client import MilvusClient

from schemas.requests.embedding_create import EmbeddingCreate
from schemas.requests.embedding_from_image_create import EmbeddingFromImageCreate
from schemas.responses.embedding_response import EmbeddingResponse
from .embedding_extractor import EmbeddingExtractor

# Configura o Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService(MilvusClient):
    """Implementa operações CRUD"""

    def __init__(self, collection_name: str = "libras_embeddings"):
        super().__init__()
        self.collection = self.get_collection(collection_name)
        self.embedding_extractor = EmbeddingExtractor("models/resnet50.h5")

    def create_embedding(self, data: EmbeddingCreate) -> EmbeddingResponse:
        """Valida e cria embedding com metadados"""
        if len(data.embedding) != 128:
            raise ValueError("Dimensão do vetor inválida")

        # Insere no Milvus
        result = self.collection.insert([{
            "embedding": data.embedding,
            "label": data.label,
            "source": data.source
        }])
        logger.info(f"Embedding criado com ID: {result.primary_keys[0]}")

        return EmbeddingResponse(
            id=result.primary_keys[0],
            label=data.label,
            status="created"
        )

    def create_embedding_from_image(self, data: EmbeddingFromImageCreate) -> EmbeddingResponse:
        """Cria embedding a partir de uma imagem"""
        logger.info(f"Extraindo embedding de {data.image_path}")
        extracted_embedding = self.embedding_extractor.extract_embedding(data.image_path)

        embedding_create_data = EmbeddingCreate(embedding=extracted_embedding,
                               label=data.label,
                               source=data.source)

        return self.create_embedding(embedding_create_data)
