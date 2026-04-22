import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(..., description="OpenAI API key")
    rabbitmq_url: str = Field("amqp://guest:guest@localhost:5672/", description="RabbitMQ broker URL")
    upload_dir: str | None = Field(None, description="Directory for upload temp files (None = system temp)")


settings = Settings()

os.environ["OPENAI_API_KEY"] = settings.openai_api_key