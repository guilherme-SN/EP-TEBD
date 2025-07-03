from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.endpoints import router

app = FastAPI(
    title="LIBRAS CBIR API",
    description="API para busca de imagens similares de LIBRAS",
    version="1.0.0"
)

origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
app.mount("/static", StaticFiles(directory="data"), name="static")


@app.get("/", tags=["Root"])
async def read_root():
    """Endpoint raiz para verificar se a API está no ar"""
    return {"message": "Hello World"}
