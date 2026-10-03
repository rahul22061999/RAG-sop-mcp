import pytest
from cache import AnswerCache


def test_key_ignores_case_and_spacing_but_not_mode_topk_or_model():
    k = AnswerCache.key
    base = k("stream", "How do I receive goods?", 5, "m")
    assert base == k("stream", "  how do  I   RECEIVE goods? ", 5, "m")
    assert base != k("tool", "How do I receive goods?", 5, "m")
    assert base != k("stream", "How do I receive goods?", 3, "m")
    assert base != k("stream", "How do I receive goods?", 5, "other-model")


class _FakeRedis:
    def __init__(self):
        self.store, self.last_ex = {}, None

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key], self.last_ex = value, ex


class _BrokenRedis:
    async def get(self, key):
        raise ConnectionError("redis down")

    async def set(self, key, value, ex=None):
        raise ConnectionError("redis down")


def _cache(client, ttl=300):
    cache = AnswerCache("redis://localhost:6379/0", ttl)
    cache._client = client
    return cache


@pytest.mark.asyncio
async def test_set_uses_the_ttl_and_get_returns_value():
    client = _FakeRedis()
    cache = _cache(client, ttl=300)
    await cache.set("k", "answer")
    assert client.last_ex == 300
    assert await cache.get("k") == "answer"


@pytest.mark.asyncio
async def test_redis_failure_is_a_miss_not_an_error_and_pauses_the_cache():
    cache = _cache(_BrokenRedis())
    assert await cache.get("k") is None
    await cache.set("k", "v")
    assert not cache._usable()


@pytest.mark.asyncio
async def test_disabled_cache_does_nothing():
    client = _FakeRedis()
    cache = _cache(client)
    cache.enabled = False
    await cache.set("k", "v")
    assert client.store == {} and await cache.get("k") is None
