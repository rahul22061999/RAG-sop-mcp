import json

import pytest
from config.settings import settings
from tools.sop_query_tool import sop_query_tool


class _FakeNode:
    def __init__(self, content: str, metadata: dict):
        self._content = content
        self.metadata = metadata

    def get_content(self) -> str:
        return self._content


class _FakeNodeWithScore:

    def __init__(self, content: str, metadata: dict, score: float):
        self.node = _FakeNode(content, metadata)
        self.score = score


class _FakeRetriever:
    def __init__(self, results):
        self._results = results

    async def aretrieve(self, query: str):
        return self._results


class _FakeIndex:
    def __init__(self, retriever):
        self._retriever = retriever

    def as_retriever(self, **kwargs):
        return self._retriever


class _FakeSOPResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self) -> dict:
        return self._payload


@pytest.fixture
def fake_index(mocker):

    def _install(retriever):
        mocker.patch.dict(settings.__dict__, {"index": _FakeIndex(retriever)})

    return _install


@pytest.mark.asyncio
async def test_flattens_nodes_into_json_serializable_dicts(mocker, fake_index):
    node = _FakeNodeWithScore(
        content="Contract Management Chief double checks payment amount...",
        metadata={"covered_pages": "35", "document_title": "Basic Payments"},
        score=0.87,
    )
    fake_index(_FakeRetriever([node]))

    captured = {}

    async def fake_generate_sop_context(question, data):
        captured["question"] = question
        captured["data"] = data
        return _FakeSOPResponse(
            {"answer": "ok", "citations": ["35"], "confidence": 1.0}
        )

    mocker.patch(
        "tools.sop_query_tool.generate_sop_context",
        side_effect=fake_generate_sop_context,
    )

    result = await sop_query_tool("What must the CMC do before payment?", top_records=3)

    assert result == {"answer": "ok", "citations": ["35"], "confidence": 1.0}

    json.dumps(captured["data"])

    assert captured["data"] == [
        {
            "text": "Contract Management Chief double checks payment amount...",
            "pages": "35",
            "title": "Basic Payments",
            "score": 0.87,
        }
    ]


@pytest.mark.asyncio
async def test_retrieval_failure_returns_safe_error_string(mocker, fake_index):
    class _BrokenRetriever:
        async def aretrieve(self, query: str):
            raise RuntimeError("connection refused")

    fake_index(_BrokenRetriever())

    result = await sop_query_tool("any question")

    assert isinstance(result, str)
    assert "Search failed" in result
    assert "connection refused" in result


@pytest.mark.asyncio
async def test_empty_retrieval_still_reaches_generator(mocker, fake_index):
    fake_index(_FakeRetriever([]))

    async def fake_generate_sop_context(question, data):
        assert data == []
        return _FakeSOPResponse(
            {"answer": "not found", "citations": [], "confidence": 0.0}
        )

    mocker.patch(
        "tools.sop_query_tool.generate_sop_context",
        side_effect=fake_generate_sop_context,
    )

    result = await sop_query_tool("obscure question with no matches")

    assert result["answer"] == "not found"
