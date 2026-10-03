import httpx
import pytest
from resilience import backoff_delay, is_transient, retry_async, retry_stream


class _Status(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


def test_backoff_is_full_jitter_and_capped():
    for attempt, ceiling in [(1, 0.5), (2, 1.0), (3, 2.0), (4, 4.0), (5, 8.0), (9, 8.0)]:
        delays = [backoff_delay(attempt, base=0.5, cap=8.0) for _ in range(200)]
        assert all(0 <= d <= ceiling for d in delays)
        assert max(delays) > ceiling * 0.5


def test_is_transient():
    assert is_transient(ConnectionRefusedError())
    assert is_transient(httpx.ConnectError("boom"))
    assert is_transient(_Status(503)) and is_transient(_Status(429))
    assert not is_transient(_Status(400))
    assert not is_transient(ValueError("bad input"))


@pytest.fixture
def no_sleep(mocker):
    return mocker.patch("resilience.asyncio.sleep", new=mocker.AsyncMock())


@pytest.mark.asyncio
async def test_retries_transient_errors_then_succeeds(no_sleep):
    calls, retries = [], []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("down")
        return "ok"

    result = await retry_async(
        flaky, attempts=4, base=0.5, cap=8, on_retry=lambda a, d, e: retries.append((a, d))
    )
    assert result == "ok" and len(calls) == 3
    assert [a for a, _ in retries] == [1, 2]
    assert no_sleep.await_count == 2


@pytest.mark.asyncio
async def test_gives_up_after_max_attempts(no_sleep):
    calls = []

    async def always_down():
        calls.append(1)
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        await retry_async(always_down, attempts=3, base=0.1, cap=1)
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_does_not_retry_non_transient_errors(no_sleep):
    calls = []

    async def bug():
        calls.append(1)
        raise ValueError("bad")

    with pytest.raises(ValueError):
        await retry_async(bug, attempts=5, base=0.1, cap=1)
    assert len(calls) == 1 and no_sleep.await_count == 0


@pytest.mark.asyncio
async def test_stream_retries_before_first_token(no_sleep):
    attempts = []

    async def stream():
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("down before any output")
        yield "a"
        yield "b"

    out = [t async for t in retry_stream(stream, attempts=3, base=0.1, cap=1)]
    assert out == ["a", "b"] and len(attempts) == 2


@pytest.mark.asyncio
async def test_stream_does_not_retry_after_output_started(no_sleep):
    attempts = []

    async def stream():
        attempts.append(1)
        yield "partial"
        raise ConnectionError("died mid-answer")

    got = []
    with pytest.raises(ConnectionError):
        async for t in retry_stream(stream, attempts=3, base=0.1, cap=1):
            got.append(t)
    assert got == ["partial"] and len(attempts) == 1
