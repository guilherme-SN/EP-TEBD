import logging
import os
import uuid
from typing import Any

from fastapi import UploadFile

from milvus.client import MilvusClient
from schemas.requests.embedding_create import EmbeddingCreate
from schemas.responses.embedding_response import EmbeddingResponse
from schemas.responses.embedding_search_response import EmbeddingSearchResponse
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

    async def create_embedding_from_image(self, image_file: UploadFile, label: str, source: str) -> EmbeddingResponse:
        """Cria embedding a partir de uma imagem"""
        temporary_directory = "data/temp_uploads"
        os.makedirs(temporary_directory, exist_ok=True)

        unique_filename = f"{uuid.uuid4()}_{image_file.filename}"
        temporary_image_path = os.path.join(temporary_directory, unique_filename)

        try:
            with open(temporary_image_path, "wb") as buffer:
                content = await image_file.read()
                buffer.write(content)

            logger.info(f"Extraindo embedding de {temporary_image_path}")
            extracted_embedding = self.embedding_extractor.extract_embedding(temporary_image_path)

            embedding_create_data = EmbeddingCreate(embedding=extracted_embedding,
                                                    label=label,
                                                    source=source)

            return self.create_embedding(embedding_create_data)
        finally:
            if os.path.exists(temporary_image_path):
                os.remove(temporary_image_path)

    async def search_top_k_most_similar(self, image_file: UploadFile, top_k: int) -> EmbeddingSearchResponse:
        """Busca pelos K embeddings mais similares com base no embedding de entrada"""
        temporary_directory = "data/temp_uploads"
        os.makedirs(temporary_directory, exist_ok=True)

        unique_filename = f"{uuid.uuid4()}_{image_file.filename}"
        temporary_image_path = os.path.join(temporary_directory, unique_filename)

        try:
            with open(temporary_image_path, "wb") as buffer:
                content = await image_file.read()
                buffer.write(content)

            extracted_embedding = self.embedding_extractor.extract_embedding(temporary_image_path)

            self.collection.load()
            logger.info(f"Coleção {self.collection.name} carregada em memória para busca")

            search_parameters = {
                "metric_type": "L2",        # Distância euclidiana como métrica de similaridade
                "params": {"nprobe": 10}    # Quantidade de clusters para verificar
            }

            search_results = self.collection.search(
                data=[extracted_embedding],             # Embedding de consulta
                anns_field="embedding",                 # Nome do campo para comparar os embeddings
                param=search_parameters,                # Parâmetros específicos para busca
                limit=top_k,                            # Retorna apenas os top K mais similares
                output_fields=["label"]                 # Metadados de retorno
            )

            results_list = []
            for result in search_results[0]:
                results_list.append(
                    EmbeddingResponse(
                        id=result.id,
                        label=result.entity.get('label'),
                        status="found",
                        distance=result.distance
                    )
                )

            self.collection.release()
            logger.info(f"Coleção {self.collection.name} liberada da memória")
            logger.info("Busca finalizada")

            return EmbeddingSearchResponse(similar_embeddings=results_list)
        finally:
            if os.path.exists(temporary_image_path):
                os.remove(temporary_image_path)

    async def delete_by_id(self, embedding_id: int) -> dict[str, str | Any] | None:
        """Deleta a entidade da coleção pelo ID"""
        if not embedding_id:
            return None

        expression = f"id == {embedding_id}"

        self.collection.load()
        logger.info(f"Coleção {self.collection.name} carregada em memória para busca")

        try:
            logger.info(f"Deletando entidades em que {expression}")
            delete_result = self.collection.delete(expression)

            self.collection.flush()

            deleted_ids = delete_result.delete_count
            logger.info(f"Total de entidade deletadas: {deleted_ids}")

            return {
                "deleted_count": deleted_ids,
                "expression": expression,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Erro ao excluir entidades: {e}")
            raise e
        finally:
            self.collection.release()
            logger.info(f"Coleção {self.collection.name} liberada da memória")
