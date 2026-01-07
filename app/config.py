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

    # Google Drive settings (optional)
    google_credentials_path: str | None = Field(
        default=None,
        alias="GOOGLE_CREDENTIALS_PATH",
    )
    google_drive_folder_id: str | None = Field(
        default=None,
        alias="GOOGLE_DRIVE_FOLDER_ID",
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
