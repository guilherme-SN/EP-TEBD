from milvus.client import MilvusClient
from schemas.requests.embedding_create import EmbeddingCreate
from schemas.responses.embedding_response import EmbeddingResponse

class EmbeddingService(MilvusClient):
    """Implementa operações CRUD"""

    def __init__(self, collection_name: str = "libras_embeddings"):
        super().__init__()
        self.collection = self.get_collection(collection_name)

    def create_embedding(self, data: EmbeddingCreate) -> EmbeddingResponse:
        """Valida e cria embedding com metadados"""
        # Pré-processamento específico
        if len(data.embedding) != 128:
            raise ValueError("Dimensão do vetor inválida")

        # Insere no Milvus
        result = self.collection.insert([{
            "embedding": data.embedding,
            "label": data.label,
            "source": data.source
        }])

        return EmbeddingResponse(
            id=result.primary_keys[0],
            label=data.label,
            status="created"
        )