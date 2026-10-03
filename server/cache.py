import hashlib
import logging
import time

import redis.asyncio as redis
from config.settings import settings

logger = logging.getLogger(__name__)

_COOLDOWN_SECONDS = 30


class AnswerCache:
    def __init__(self, url: str, ttl_seconds: int, enabled: bool = True) -> None:
        self.ttl = ttl_seconds
        self.enabled = enabled
        self._down_until = 0.0
        self._client = redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=0.25,
            socket_timeout=0.25,
        )

    @staticmethod
    def key(mode: str, query: str, top_k: int, model: str) -> str:
        normalized = " ".join(query.lower().split())
        digest = hashlib.sha256(f"{model}|{top_k}|{normalized}".encode()).hexdigest()
        return f"wms_sop:answer:{mode}:{digest[:32]}"

    def _usable(self) -> bool:
        return self.enabled and time.monotonic() >= self._down_until

    def _failed(self, action: str, exc: Exception) -> None:
        self._down_until = time.monotonic() + _COOLDOWN_SECONDS
        logger.warning("Redis %s failed, skipping cache for %ss: %s", action, _COOLDOWN_SECONDS, exc)

    async def get(self, key: str) -> str | None:
        if not self._usable():
            return None
        try:
            return await self._client.get(key)
        except Exception as exc:
            self._failed("get", exc)
            return None

    async def set(self, key: str, value: str) -> None:
        if not self._usable():
            return
        try:
            await self._client.set(key, value, ex=self.ttl)
        except Exception as exc:
            self._failed("set", exc)


answer_cache = AnswerCache(
    settings.redis_url, settings.cache_ttl_seconds, settings.cache_enabled
)
