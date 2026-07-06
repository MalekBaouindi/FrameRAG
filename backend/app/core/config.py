from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "FrameworkRAG"
    environment: str = "development"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "langchain-latest"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    openai_api_key: str | None = None
    mistral_api_key: str | None = None
    deepseek_api_key: str | None = None

    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "prithivida/Splade_PP_en_v1"
    rerank_model: str = "mixedbread-ai/mxbai-rerank-xsmall-v1"
    top_k: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
