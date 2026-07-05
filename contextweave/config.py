"""Configuration for ContextWeave."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    groq_api_key: str = ""
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reasoning_model: str = "llama-3.1-8b-instant"
    extraction_model: str = "llama-3.1-8b-instant"
    transcription_model: str = "whisper-large-v3-turbo"

    chroma_persist_dir: str = "./chroma_data"
    sqlite_db_path: str = "./contextweave.db"
    data_dir: str = "./data"

    # Postgres + pgvector (e.g. Supabase). When set, replaces SQLite,
    # ChromaDB, and the local user registry with one external database.
    database_url: str = ""
    # Supabase Auth: project JWT secret; enables sign-in tokens alongside cw_ keys
    supabase_jwt_secret: str = ""
    # Supabase Auth for the web UI: project URL + anon key are public by design
    # and are served to the browser so it can run the sign-in flow itself.
    supabase_url: str = ""
    supabase_anon_key: str = ""

    digest_cache_hours: float = 12.0

    # Pushed daily digest: any SMTP provider (Gmail app password, Resend,
    # Mailgun, …). Delivery is off until smtp_host is set.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    digest_from_email: str = ""  # defaults to smtp_username when empty
    # Absolute origin used to build unsubscribe links in emails
    public_base_url: str = ""

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
