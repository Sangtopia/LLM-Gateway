from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from gateway.models import DEFAULT_FALLBACKS, DEFAULT_REFUSAL_FALLBACKS
from gateway.routing import DEFAULT_ALIASES, Alias


class Settings(BaseSettings):
    # hide_input_in_errors: settings hold secrets, and a validation error would otherwise print them.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", hide_input_in_errors=True)

    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    database_url: str = "sqlite+aiosqlite:///./gateway.db"
    redis_url: str = "redis://localhost:6379/0"

    upstream_connect_timeout_s: float = 5.0
    upstream_read_timeout_s: float = 600.0  # Whole response, non-streaming. Long generations take minutes.
    upstream_stream_idle_timeout_s: float = 60.0  # Max gap between stream events; Anthropic sends pings.
    upstream_max_retries: int = 2
    # Connections to Anthropic. httpx's default keeps only 20 idle, so above 20 concurrent
    # requests connections are constantly closed and reopened.
    upstream_max_connections: int = 500
    upstream_max_keepalive: int = 200
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


    @field_validator("redis_url")
    @classmethod
    def _clean_redis_url(cls, url: str) -> str:
        """Tolerate stray whitespace or quotes from copy-paste; fail clearly (without echoing the
        password) if the value isn't a Redis URL, e.g. when copied with `redis-cli -u` in front."""
        url = url.strip().strip("'\"").strip()
        if not url.startswith(("redis://", "rediss://", "unix://")):
            raise ValueError("REDIS_URL must start with redis:// or rediss://")
        return url

    @field_validator("database_url")
    @classmethod
    def _async_driver(cls, url: str) -> str:
        """Hosts like Render hand out postgres:// or postgresql:// URLs; the gateway needs the async driver."""
        for prefix in ("postgres://", "postgresql://"):
            if url.startswith(prefix):
                return "postgresql+asyncpg://" + url[len(prefix):]
        return url


settings = Settings()
