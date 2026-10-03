import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TypeVar

import httpx
from sqlalchemy.exc import OperationalError

T = TypeVar("T")
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (OSError, httpx.TransportError, OperationalError)):
        return True
    return getattr(exc, "status_code", None) in RETRYABLE_STATUS


def backoff_delay(attempt: int, base: float, cap: float) -> float:
    return random.uniform(0, min(cap, base * 2 ** (attempt - 1)))


async def retry_async(
    factory: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    base: float,
    cap: float,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
) -> T:
    for attempt in range(1, attempts + 1):
        try:
            return await factory()
        except Exception as exc:
            if attempt == attempts or not is_transient(exc):
                raise
            delay = backoff_delay(attempt, base, cap)
            if on_retry:
                on_retry(attempt, delay, exc)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def retry_stream(
    factory: Callable[[], AsyncIterator[T]],
    *,
    attempts: int,
    base: float,
    cap: float,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
) -> AsyncIterator[T]:
    for attempt in range(1, attempts + 1):
        started = False
        try:
            async for item in factory():
                started = True
                yield item
            return
        except Exception as exc:
            if started or attempt == attempts or not is_transient(exc):
                raise
            delay = backoff_delay(attempt, base, cap)
            if on_retry:
                on_retry(attempt, delay, exc)
            await asyncio.sleep(delay)
