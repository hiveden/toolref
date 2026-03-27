"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """ToolRef application settings.

    All values are read from environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Application ---
    app_name: str = "ToolRef"
    app_version: str = "0.1.0"
    debug: bool = False

    # --- PostgreSQL ---
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str = "toolref"
    postgres_password: str = "toolref_secret"
    postgres_db: str = "toolref"

    @property
    def database_url(self) -> str:
        """Async database connection URL."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # --- Redis ---
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0

    @property
    def redis_url(self) -> str:
        """Redis connection URL."""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # --- Milvus ---
    milvus_host: str = "milvus-standalone"
    milvus_port: int = 19530

    # --- MinIO ---
    minio_host: str = "minio"
    minio_port: int = 9000
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "toolref-documents"
    minio_secure: bool = False

    # --- Embedding ---
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 32

    # --- Chunking ---
    chunk_parent_size: int = 1024
    chunk_parent_overlap: int = 64
    chunk_child_size: int = 256
    chunk_child_overlap: int = 32

    # --- LLM ---
    llm_provider: str = "ollama"  # "ollama" | "deepseek" | "openai"
    llm_model: str = "qwen2.5:7b"
    llm_temperature: float = 0.1
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"
    deepseek_api_key: str = ""
    deepseek_api_base: str = "https://api.deepseek.com/v1"

    # --- Reranker ---
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = 5

    # --- Semantic Cache ---
    cache_similarity_threshold: float = 0.92
    cache_default_ttl: int = 86400       # 24 hours
    cache_hot_ttl: int = 259200          # 72 hours
    cache_low_freq_ttl: int = 43200      # 12 hours

    # --- RAG Retrieval ---
    grading_relevance_threshold: float = 0.6
    max_rewrite_count: int = 2
    consistency_check_enabled: bool = False  # MVP: disabled

    # --- CORS ---
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]


settings = Settings()
