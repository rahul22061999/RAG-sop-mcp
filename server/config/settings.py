from functools import cached_property
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    service_name: str = "wms-sop-server"
    service_description: str = (
        "Production-grade RAG WMS MCP service"
    )

    port: int = 8001
    log_level: str = "INFO"
    log_file: str = "logs/wms-sop-mcp.log"

    # Postgres (pgvector) — Google Cloud SQL / AlloyDB or any Postgres
    # instance with the pgvector extension enabled.
    pg_host: str
    pg_port: int = 5432
    pg_database: str
    pg_user: str
    pg_password: SecretStr
    pg_table_name: str = "sop_chunks"
    pg_schema_name: str = "public"
    pg_embed_dim: int = 1536
    pg_ssl_mode: str = "disable"

    openai_api_key: SecretStr
    openai_embedding_model: str

    # unkey cred
    unkey_root_api_key: SecretStr

    # Ollama endpoint used for answer generation (server/tools/rag_generator.py).
    # "http://host.docker.internal:11434" is correct for Docker Desktop on
    # Mac/Windows only - it does not resolve on Linux hosts or most managed
    # container platforms. Override via env for any other deployment target.
    ollama_model: str = "gemma4:31b-cloud"
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_request_timeout: float = 60.0

    @cached_property
    def embed_model(self) -> OpenAIEmbedding:
        return OpenAIEmbedding(
            model=self.openai_embedding_model,
            api_key=self.openai_api_key.get_secret_value(),
        )

    @cached_property
    def vector_store(self) -> PGVectorStore:

        base_url = URL.create(
            drivername="postgresql",
            username=self.pg_user,
            password=self.pg_password.get_secret_value(),
            host=self.pg_host,
            port=self.pg_port,
            database=self.pg_database,
        )
        connection_string = base_url.set(
            drivername="postgresql+psycopg2",
            query={"sslmode": self.pg_ssl_mode},
        ).render_as_string(hide_password=False)
        async_connection_string = base_url.set(
            drivername="postgresql+asyncpg",
            query=(
                {}
                if self.pg_ssl_mode == "disable"
                else {"ssl": self.pg_ssl_mode}
            ),
        ).render_as_string(hide_password=False)

        return PGVectorStore(
            connection_string=connection_string,
            async_connection_string=async_connection_string,
            table_name=self.pg_table_name,
            schema_name=self.pg_schema_name,
            embed_dim=self.pg_embed_dim,
            hybrid_search=True,
            text_search_config="english",
        )

    @cached_property
    def index(self) -> VectorStoreIndex:
        return VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model,
        )


settings = Settings()
