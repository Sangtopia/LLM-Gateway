from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    database_url: str = "sqlite+aiosqlite:///./gateway.db"
    redis_url: str = "redis://localhost:6379/0"

    upstream_connect_timeout_s: float = 5.0
    upstream_read_timeout_s: float = 600.0  # Whole response, non-streaming. Long generations take minutes.
    upstream_stream_idle_timeout_s: float = 60.0  # Max gap between stream events; Anthropic sends pings.
    upstream_max_retries: int = 2


settings = Settings()
