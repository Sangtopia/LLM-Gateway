from fastapi import FastAPI

from gateway.config import settings

app = FastAPI(title="LLM Gateway")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "upstream_key_configured": bool(settings.anthropic_api_key)}
