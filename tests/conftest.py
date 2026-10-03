import pytest
from cache import answer_cache


@pytest.fixture(autouse=True)
def _no_real_cache(monkeypatch):
    monkeypatch.setattr(answer_cache, "enabled", False)
