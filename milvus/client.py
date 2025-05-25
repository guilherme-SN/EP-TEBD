from pymilvus import connections, utility, Collection
from config.settings import MILVUS_CONFIG


class MilvusClient:
    """Classe de conexão com o Milvus"""

    def __init__(self):
        connections.connect(
            alias="default",
            host=MILVUS_CONFIG['host'],
            port=MILVUS_CONFIG['port']
        )

    def get_collection(self, collection_name: str):
        """Retorna uma coleção"""
        if not utility.has_collection(collection_name):
            raise ValueError(f"Coleção {collection_name} não existe")
        collection = Collection(collection_name)
        collection.load()

        return collection