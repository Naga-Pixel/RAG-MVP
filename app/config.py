from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Settings for b_rag.
    
    All settings can be configured via environment variables or .env file.
    """

    # Read from .env and ignore unknown vars
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI settings
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )
    openai_chat_model: str = Field(
        default="gpt-4-turbo-preview",
        alias="OPENAI_CHAT_MODEL",
    )

    # Qdrant settings
    qdrant_url: str = Field(
        default="http://localhost:6333",
        alias="QDRANT_URL",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        alias="QDRANT_API_KEY",
    )
    qdrant_collection: str = Field(
        default="documents_default",
        alias="QDRANT_COLLECTION",
    )

    # Ingestion settings
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    retrieval_limit: int = Field(default=12, alias="RETRIEVAL_LIMIT")

    # Retrieval and reranking settings
    retrieve_k: int = Field(
        default=12,
        alias="RETRIEVE_K",
        description="Number of candidates to retrieve from vector search",
    )
    final_k: int = Field(
        default=6,
        alias="FINAL_K",
        description="Number of chunks to include in final context after reranking",
    )
    rerank_enabled: bool = Field(
        default=False,
        alias="RERANK_ENABLED",
        description="Enable reranking of retrieved candidates",
    )
    rerank_k: int | None = Field(
        default=None,
        alias="RERANK_K",
        description="Number of candidates to rerank (defaults to RETRIEVE_K)",
    )
    max_chunks_per_doc: int = Field(
        default=3,
        alias="MAX_CHUNKS_PER_DOC",
        description="Maximum chunks from same document in final results (diversity)",
    )

    # FTS shadow index settings
    # NOTE: This flag affects WRITES ONLY (ingestion).
    # It does NOT affect retrieval, /ask, or any read path.
    # When enabled, chunk text is shadow-written to Postgres for future hybrid search.
    fts_shadow_enabled: bool = Field(
        default=False,
        alias="FTS_SHADOW_ENABLED",
        description="Enable shadow Postgres FTS index during ingestion (Phase A, writes only)",
    )
    fts_shadow_schema: str = Field(
        default="internal",
        alias="FTS_SHADOW_SCHEMA",
        description="Postgres schema for FTS shadow table",
    )
    fts_shadow_table: str = Field(
        default="chunks",
        alias="FTS_SHADOW_TABLE",
        description="Postgres table name for FTS shadow chunks",
    )

    # Multi-tenancy
    default_tenant_id: str = Field(default="default", alias="DEFAULT_TENANT_ID")

    # Google Drive settings (optional - for service account auth)
    google_credentials_path: str | None = Field(
        default=None,
        alias="GOOGLE_CREDENTIALS_PATH",
    )
    google_drive_folder_id: str | None = Field(
        default=None,
        alias="GOOGLE_DRIVE_FOLDER_ID",
    )

    # Google Drive OAuth settings (for user-based auth with Picker)
    google_drive_client_id: str | None = Field(
        default=None,
        alias="GOOGLE_DRIVE_CLIENT_ID",
    )
    google_drive_client_secret: str | None = Field(
        default=None,
        alias="GOOGLE_DRIVE_CLIENT_SECRET",
    )
    google_drive_redirect_uri: str = Field(
        default="http://localhost:8000/oauth/google/drive/callback",
        alias="GOOGLE_DRIVE_REDIRECT_URI",
    )
    google_picker_api_key: str | None = Field(
        default=None,
        alias="GOOGLE_PICKER_API_KEY",
    )
    drive_token_encryption_key: str | None = Field(
        default=None,
        alias="DRIVE_TOKEN_ENCRYPTION_KEY",
        description="Fernet key for encrypting Drive refresh tokens",
    )

    # Database (Postgres via Supabase or direct connection)
    database_url: str | None = Field(
        default=None,
        alias="DATABASE_URL",
        description="Postgres connection string for Drive OAuth data",
    )

    # Supabase settings (for frontend auth)
    supabase_url: str | None = Field(
        default=None,
        alias="SUPABASE_URL",
    )
    supabase_anon_key: str | None = Field(
        default=None,
        alias="SUPABASE_ANON_KEY",
    )
    supabase_jwt_secret: str | None = Field(
        default=None,
        alias="SUPABASE_JWT_SECRET",
        description="Supabase JWT secret for verifying access tokens",
    )

    # API Security
    api_key: str | None = Field(
        default=None,
        alias="API_KEY",
        description="API key for authenticating sync/admin endpoints",
    )
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        alias="CORS_ORIGINS",
        description="Comma-separated list of allowed CORS origins",
    )
    rate_limit_ask: str = Field(
        default="10/minute",
        alias="RATE_LIMIT_ASK",
        description="Rate limit for /ask endpoint (e.g., 10/minute)",
    )
    rate_limit_sync: str = Field(
        default="5/minute",
        alias="RATE_LIMIT_SYNC",
        description="Rate limit for sync endpoints (e.g., 5/minute)",
    )
    allowed_sync_directories: list[str] = Field(
        default_factory=list,
        alias="ALLOWED_SYNC_DIRECTORIES",
        description="Comma-separated list of allowed directories for local sync",
    )
    credentials_directory: str = Field(
        default="credentials",
        alias="CREDENTIALS_DIRECTORY",
        description="Directory where credentials files must be located",
    )

    # ---- Compatibility properties ----

    @property
    def embedding_model(self) -> str:
        return self.openai_embedding_model

    @property
    def chat_model(self) -> str:
        return self.openai_chat_model

    # UPPERCASE versions for any old references
    @property
    def OPENAI_API_KEY(self) -> str:
        return self.openai_api_key

    @property
    def OPENAI_EMBEDDING_MODEL(self) -> str:
        return self.openai_embedding_model

    @property
    def OPENAI_CHAT_MODEL(self) -> str:
        return self.openai_chat_model

    @property
    def QDRANT_URL(self) -> str:
        return self.qdrant_url

    @property
    def QDRANT_API_KEY(self) -> str | None:
        return self.qdrant_api_key

    @property
    def QDRANT_COLLECTION(self) -> str:
        return self.qdrant_collection

    @property
    def CHUNK_SIZE(self) -> int:
        return self.chunk_size

    @property
    def CHUNK_OVERLAP(self) -> int:
        return self.chunk_overlap

    @property
    def RETRIEVAL_LIMIT(self) -> int:
        return self.retrieval_limit

    @property
    def DEFAULT_TENANT_ID(self) -> str:
        return self.default_tenant_id


settings = Settings()
