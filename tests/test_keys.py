import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from gateway.db import api_keys_table, init_db
from gateway.keys import KeyStore, extract_key, hash_key


def run_with_store(tmp_path, fn):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'keys.db'}")
        await init_db(engine)
        try:
            return await fn(KeyStore(engine), engine)
        finally:
            await engine.dispose()

    return asyncio.run(run())


def test_create_then_authenticate(tmp_path):
    async def fn(store, engine):
        key, plaintext = await store.create("helpdesk", rpm_limit=60, monthly_budget_usd=20.0)
        found = await store.authenticate(plaintext)
        return key, plaintext, found

    key, plaintext, found = run_with_store(tmp_path, fn)
    assert plaintext.startswith("gw-") and len(plaintext) > 40
    assert found == key
    assert found.rpm_limit == 60 and found.tpm_limit is None and found.monthly_budget_usd == 20.0


def test_only_the_hash_is_stored(tmp_path):
    async def fn(store, engine):
        _, plaintext = await store.create("helpdesk")
        async with engine.connect() as conn:
            row = (await conn.execute(select(api_keys_table))).mappings().one()
        return plaintext, dict(row)

    plaintext, row = run_with_store(tmp_path, fn)
    assert row["key_hash"] == hash_key(plaintext)
    assert plaintext not in [str(v) for v in row.values()]
    assert row["key_prefix"] == plaintext[:10]


def test_unknown_malformed_and_revoked_keys_are_rejected(tmp_path):
    async def fn(store, engine):
        key, plaintext = await store.create("helpdesk")
        results = [
            await store.authenticate(None),
            await store.authenticate(""),
            await store.authenticate("gw-not-a-real-key"),
            await store.authenticate("sk-ant-something"),
        ]
        assert await store.revoke(key.id)
        assert not await store.revoke(key.id)  # Already revoked.
        results.append(await store.authenticate(plaintext))
        listed = await store.list()
        return results, listed

    results, listed = run_with_store(tmp_path, fn)
    assert results == [None] * 5
    assert listed[0]["revoked_at"] is not None
    assert "key_hash" not in listed[0]


def test_extract_key_from_headers():
    assert extract_key({"x-api-key": " gw-a "}) == "gw-a"
    assert extract_key({"authorization": "Bearer gw-b"}) == "gw-b"
    assert extract_key({"x-api-key": "gw-a", "authorization": "Bearer gw-b"}) == "gw-a"
    assert extract_key({}) is None


def test_revoke_in_another_process_takes_effect_when_cache_expires(tmp_path):
    async def fn(store, engine):
        _, plaintext = await store.create("helpdesk")
        assert await store.authenticate(plaintext)  # Now cached.
        other_process = KeyStore(engine)
        key_id = (await other_process.list())[0]["id"]
        await other_process.revoke(key_id)
        still_cached = await store.authenticate(plaintext)
        store._cache_ttl = 0
        store._cache.clear()
        after_expiry = await store.authenticate(plaintext)
        return still_cached, after_expiry

    still_cached, after_expiry = run_with_store(tmp_path, fn)
    assert still_cached is not None
    assert after_expiry is None
