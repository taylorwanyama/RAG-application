from enum import Enum
from pydantic_settings import BaseSettings, SettingsConfigDict

class Environment(Enum):
    DEVELOPMENT = 'development'
    STAGING = 'staging'
    PRODUCTION = 'production'


class Settings(BaseSettings):

    groq_api_key: str
    pinecone_api_key: str
    pinecone_index: str

    groq_model: str = "openai/gpt-oss-20b"
    groq_timeout: float = 30.0

    environment: Environment = Environment.DEVELOPMENT

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()