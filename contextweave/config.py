"""Configuration for ContextWeave."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    groq_api_key: str = ""
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reasoning_model: str = "llama-3.1-8b-instant"
    extraction_model: str = "llama-3.1-8b-instant"

    chroma_persist_dir: str = "./chroma_data"
    sqlite_db_path: str = "./contextweave.db"

    chunk_max_tokens: int = 512
    chunk_overlap_sentences: int = 2
    embedding_dimension: int = 384

    retrieval_top_k: int = 20
    retrieval_final_k: int = 8
    graph_hop_depth: int = 2

    decay_half_life_days: float = 30.0
    access_boost_factor: float = 1.2
    connection_density_weight: float = 0.3

    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_prefix": "CW_", "env_file": ".env"}


settings = Settings()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
