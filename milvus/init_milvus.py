import logging
from pymilvus import connections, utility, Collection, FieldSchema, DataType, CollectionSchema

# Configura o Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Conecta ao Milvus
connections.connect("default", host="localhost", port="19530")

# Cria coleção
def create_collection():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),  # 128 dimensões
        FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=100)
    ]

    # Esquema da coleção
    schema = CollectionSchema(fields, description="Coleção para armazenar embeddings de libras")

    # Cria a coleção
    collection = Collection("libras_embeddings", schema)

    # Criar índice (obrigatório para buscas por vetor)
    index_params = {
        "index_type": "IVF_FLAT",   # Tipo de algoritmo de índice
        "metric_type": "L2",        # Métrica de similaridade (L2 = Distância Euclidiana)
        "params": {"nlist": 128}    # Parâmetros específicos do algoritmo
    }
    collection.create_index("embedding", index_params)
    logger.info("Coleção 'libras_embeddings' criada com sucesso")


# Verifica e cria a coleção se ela não existir
if not utility.has_collection("libras_embeddings"):
    create_collection()
else:
    logger.warning("Coleção já existe")
