"""Gateway API keys ("virtual keys").

Apps authenticate with gw-... keys issued here, never with the real Anthropic
key. Only a SHA-256 hash of each key is stored. A slow password hash like bcrypt
isn't needed: the keys are 256-bit random values, so they can't be guessed, and
hashing runs on every request.

Manage keys from the command line:

    python -m gateway.keys create helpdesk-agent [--rpm 60] [--tpm 100000] [--budget 20]
    python -m gateway.keys list
    python -m gateway.keys revoke <key id>
"""

import argparse
import asyncio
import hashlib
import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from gateway.db import api_keys_table, init_db

KEY_PREFIX = "gw-"


@dataclass(frozen=True)
class ApiKey:
    id: str
    name: str
    rpm_limit: int | None = None
    tpm_limit: int | None = None
    monthly_budget_usd: float | None = None


def hash_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode()).hexdigest()


class KeyStore:
    """Creates, looks up and revokes keys.

    Successful lookups are cached in memory for `cache_ttl` seconds so most
    requests don't touch the database. The cost: a key revoked by another process
    (e.g. the CLI) keeps working in this one for up to `cache_ttl` seconds.
    """

    def __init__(self, engine: AsyncEngine, cache_ttl: float = 30.0):
        self._engine = engine
        self._cache_ttl = cache_ttl
        self._cache: dict[str, tuple[ApiKey, float]] = {}  # key hash -> (key, expiry)

    async def create(
        self,
        name: str,
        rpm_limit: int | None = None,
        tpm_limit: int | None = None,
        monthly_budget_usd: float | None = None,
    ) -> tuple[ApiKey, str]:
        """Create a key. Returns the plaintext key, which can't be recovered later."""
        plaintext = KEY_PREFIX + secrets.token_urlsafe(32)
        key = ApiKey(
            id="key_" + secrets.token_hex(6),
            name=name,
            rpm_limit=rpm_limit,
            tpm_limit=tpm_limit,
            monthly_budget_usd=monthly_budget_usd,
        )
        async with self._engine.begin() as conn:
            await conn.execute(
                insert(api_keys_table).values(
                    id=key.id,
                    name=name,
                    key_hash=hash_key(plaintext),
                    key_prefix=plaintext[:10],
                    created_at=datetime.now(UTC),
                    rpm_limit=rpm_limit,
                    tpm_limit=tpm_limit,
                    monthly_budget_usd=monthly_budget_usd,
                )
            )
        return key, plaintext

    async def authenticate(self, plaintext: str | None) -> ApiKey | None:
        """The active key matching `plaintext`, or None if it's missing, unknown or revoked."""
        if not plaintext or not plaintext.startswith(KEY_PREFIX):
            return None
        key_hash = hash_key(plaintext)
        cached = self._cache.get(key_hash)
        if cached and cached[1] > time.monotonic():
            return cached[0]

        t = api_keys_table
        query = select(t.c.id, t.c.name, t.c.rpm_limit, t.c.tpm_limit, t.c.monthly_budget_usd).where(
            t.c.key_hash == key_hash, t.c.revoked_at.is_(None)
        )
        async with self._engine.connect() as conn:
            row = (await conn.execute(query)).mappings().first()
        if row is None:
            # Unknown keys aren't cached, so random guesses can't fill memory.
            self._cache.pop(key_hash, None)
            return None
        key = ApiKey(**row)
        self._cache[key_hash] = (key, time.monotonic() + self._cache_ttl)
        return key

    async def list(self) -> list[dict]:
        t = api_keys_table
        query = select(*[c for c in t.c if c.name != "key_hash"]).order_by(t.c.created_at)
        async with self._engine.connect() as conn:
            return [dict(row) for row in (await conn.execute(query)).mappings()]

    async def revoke(self, key_id: str) -> bool:
        t = api_keys_table
        query = update(t).where(t.c.id == key_id, t.c.revoked_at.is_(None)).values(revoked_at=datetime.now(UTC))
        async with self._engine.begin() as conn:
            revoked = (await conn.execute(query)).rowcount > 0
        self._cache = {h: entry for h, entry in self._cache.items() if entry[0].id != key_id}
        return revoked


def extract_key(headers) -> str | None:
    """The key from x-api-key, or from Authorization: Bearer (the SDKs' auth_token option)."""
    if key := headers.get("x-api-key"):
        return key.strip()
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


async def _cli(args: argparse.Namespace) -> None:
    from gateway.config import settings

    engine = create_async_engine(settings.database_url)
    await init_db(engine)
    store = KeyStore(engine)
    try:
        if args.command == "create":
            key, plaintext = await store.create(args.name, args.rpm, args.tpm, args.budget)
            print(f"Created {key.id} ({key.name})")
            print(f"\n    {plaintext}\n")
            print("Save this key now. It is stored hashed and can't be shown again.")
        elif args.command == "list":
            for k in await store.list():
                status = "revoked" if k["revoked_at"] else "active"
                limits = ", ".join(
                    f"{label}={k[col]}"
                    for label, col in [("rpm", "rpm_limit"), ("tpm", "tpm_limit"), ("budget $", "monthly_budget_usd")]
                    if k[col] is not None
                ) or "no limits"
                print(f"{k['id']}  {k['key_prefix']}...  {k['name']:<24} {status:<8} {limits}")
        elif args.command == "revoke":
            print("Revoked." if await store.revoke(args.key_id) else "No active key with that id.")
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m gateway.keys", description="Manage gateway API keys.")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="create a key")
    create.add_argument("name")
    create.add_argument("--rpm", type=int, help="requests per minute")
    create.add_argument("--tpm", type=int, help="tokens per minute")
    create.add_argument("--budget", type=float, help="monthly budget in USD")
    sub.add_parser("list", help="list keys")
    revoke = sub.add_parser("revoke", help="revoke a key")
    revoke.add_argument("key_id")
    asyncio.run(_cli(parser.parse_args()))


if __name__ == "__main__":
    main()
