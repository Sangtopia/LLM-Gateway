from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from gateway.models import DEFAULT_FALLBACKS


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
    circuit_failure_threshold: int = 5  # Consecutive failed attempts before a model is skipped.
    circuit_cooldown_s: float = 30.0  # Before letting a probe request through.

    # JSON in the environment, e.g. FALLBACK_CHAINS={"claude-sonnet-5-5": ["claude-haiku-5-5"]}
    fallback_chains: dict[str, list[str]] = Field(default_factory=lambda: dict(DEFAULT_FALLBACKS))


settings = Settings()
