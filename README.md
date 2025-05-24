# EP - TEBD

## CBIR - Libras

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
