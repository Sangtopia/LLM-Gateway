from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    database_url: str = "sqlite+aiosqlite:///./gateway.db"
    redis_url: str = "redis://localhost:6379/0"


settings = Settings()
