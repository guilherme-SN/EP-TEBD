# EP - TEBD

## CBIR - Libras

### Como rodar?
1. Baixe o docker/docker-compose
2. Para rodar o Milvus: execute o comando `docker-compose up -d` na raiz do projeto
3. Para rodar o Attu (Interface gráfica): execute o comando `docker run --name attu -p 8000:3000 -e HOST_URL=http://{IPV4}:8000 -e MILVUS_URL=http://{IPV4}:19530 zilliz/attu:v2.5.6`
   - Troque o `IPV4` pelo seu IPV4
4. Para rodar o FastAPI: execute o comando `uvicorn api.endpoints:router --host 0.0.0.0 --port 8080 --reload`
   
### Estrutura do projeto
```
EP-TEBD/  
│
├── data/                  # Dados brutos e processados
│   ├── raw/               # Imagens originais (ex.: A.jpg, B.jpg)
│   └── processed/         # Imagens pré-processadas (redimensionadas/normalizadas)
│
├── models/               # Modelos de ML e embeddings
│   ├── embeddings/       # Vetores salvos (.npy ou similares)
│   └── resnet50.h5       # Modelo pré-treinado (opcional, se for fine-tune)
│
├── milvus/               # Configuração e operações do Milvus
│   ├── client.py         # Cliente Milvus (conexão, coleções)
│   └── schemas.py        # Definição das coleções (ex.: "libras_letters")
│
├── services/             # Lógica principal
│   ├── embedding.py      # Extrai embeddings (ResNet, CNN custom, etc.)
│   └── search.py         # Busca por similaridade no Milvus
│
├── api/                  # Interface de comunicação (REST/CLI)
│   ├── app.py            # FastAPI/Flask (endpoints)
│   └── cli.py            # Scripts de linha de comando (para testes)
│
├── config/               # Configurações globais
│   └── settings.py       # Parâmetros (dimensões do embedding, paths)
│
└── README.md             # Documentação do projeto
```
