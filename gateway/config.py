from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from gateway.models import DEFAULT_FALLBACKS, DEFAULT_REFUSAL_FALLBACKS
from gateway.routing import DEFAULT_ALIASES, Alias


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
    admin_token: str = ""  # For /admin and the dashboard; empty means local requests only.
    semantic_cache: bool = False  # Needs fastembed; downloads a ~30 MB embedding model on first use.
    auto_prompt_cache: bool = True  # Add cache_control to requests that don't set it (see promptcache.py).

    # JSON in the environment, e.g. FALLBACK_CHAINS={"claude-sonnet-5-5": ["claude-haiku-5-5"]}
    fallback_chains: dict[str, list[str]] = Field(default_factory=lambda: dict(DEFAULT_FALLBACKS))
    # Retry declined (stop_reason "refusal") non-streaming requests on another model. JSON, like FALLBACK_CHAINS.
    refusal_fallbacks: dict[str, list[str]] = Field(default_factory=lambda: dict(DEFAULT_REFUSAL_FALLBACKS))
    # JSON, e.g. ROUTE_ALIASES={"cheap": {"strategy": "cheapest", "models": ["claude-haiku-5-5"]}}
    route_aliases: dict[str, Alias] = Field(default_factory=lambda: dict(DEFAULT_ALIASES))


settings = Settings()
