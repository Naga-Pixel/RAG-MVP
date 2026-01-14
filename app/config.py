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

    # API Security
    api_key: str | None = Field(
        default=None,
        alias="API_KEY",
        description="API key for authenticating sync/admin endpoints",
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
