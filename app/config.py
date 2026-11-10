import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, ValidationError

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    api_key: SecretStr
    model: str
    llm_rpm: int = Field(default=12, ge=1)

    pinecone_api_key: SecretStr
    pinecone_index: str = Field(pattern=r"^[a-z0-9-]+$")
    pinecone_namespace: str = "compras"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embedding_model: str
    embedding_dimension: int = Field(default=1536, ge=128)
    top_k: int = Field(default=4, ge=1, le=20)
    bm25_peso: float = Field(default=0.4, ge=0, le=1)

    redis_url: str = "redis://localhost:6379"
    phoenix_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_project: str = "proyecto-final"

    max_pasos: int = Field(default=8, ge=3, le=20)
    umbral_aprobacion: int = Field(default=500_000, ge=0)
    worker_concurrencia: int = Field(default=5, ge=1)
    timeout_job: int = Field(default=300, ge=10)

    catalogo: Path = BASE_DIR / "data" / "catalogo.json"
    docs_dir: Path = BASE_DIR / "data" / "docs"


ENV = {
    "api_key": "GEMINI_API_KEY",
    "model": "GEMINI_MODEL",
    "llm_rpm": "LLM_RPM",
    "pinecone_api_key": "PINECONE_API_KEY",
    "pinecone_index": "PINECONE_INDEX",
    "pinecone_namespace": "PINECONE_NAMESPACE",
    "pinecone_cloud": "PINECONE_CLOUD",
    "pinecone_region": "PINECONE_REGION",
    "embedding_model": "EMBEDDING_MODEL",
    "embedding_dimension": "EMBEDDING_DIMENSION",
    "top_k": "TOP_K",
    "bm25_peso": "BM25_PESO",
    "redis_url": "REDIS_URL",
    "phoenix_endpoint": "PHOENIX_ENDPOINT",
    "phoenix_project": "PHOENIX_PROJECT",
    "umbral_aprobacion": "UMBRAL_APROBACION",
    "worker_concurrencia": "WORKER_CONCURRENCIA",
}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(BASE_DIR / ".env")
    valores = {campo: os.getenv(var) for campo, var in ENV.items()}
    try:
        return Settings(**{k: v for k, v in valores.items() if v})
    except ValidationError as exc:
        faltan = [ENV[e["loc"][0]] for e in exc.errors() if e["type"] == "missing"]
        if faltan:
            raise RuntimeError(f"Faltan variables en el .env: {', '.join(faltan)} (ver .env.example)") from exc
        raise
