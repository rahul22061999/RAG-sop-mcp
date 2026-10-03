import asyncio
import json
import logging
from collections.abc import AsyncIterator
from functools import partial
from typing import Any

from cache import answer_cache
from config.settings import settings
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from resilience import retry_async, retry_stream
from telemetry import RagTrace
from tools.rag_generator import generate_sop_context, stream_sop_context

logger = logging.getLogger(__name__)

_RETRY = {
    "attempts": settings.retry_attempts,
    "base": settings.retry_base_delay,
    "cap": settings.retry_max_delay,
}
_REPLAY_CHARS = 48


async def _retrieve(query: str, top_records: int) -> list[dict[str, Any]]:
    top_records = max(1, min(top_records, 20))
    retriever = settings.index.as_retriever(
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
        similarity_top_k=top_records,
    )
    nodes = await retriever.aretrieve(query.strip())
    return [
        {
            "text": item.node.get_content(),
            "pages": item.node.metadata.get("covered_pages"),
            "title": item.node.metadata.get("document_title"),
            "score": item.score,
        }
        for item in nodes
    ]


async def sop_query_tool(
    query: str,
    top_records: int = 5,
) -> dict[str, Any] | str:
    """
    Search the WMS SOP knowledge base using Postgres pgvector hybrid retrieval.
    Hybrid retrieval combines:
    - keyword/full-text search (Postgres tsvector)
    - vector similarity search (pgvector)
    Args:
        query:
            Natural-language question or search phrase.
        top_records:
            Maximum number of matching SOP chunks to return.
            Must be between 1 and 20.
    Returns:
        The grounded answer as a dict (answer, citations, confidence), or an
        error string. Non-streaming; used by the MCP server.
    """
    logger.info(f"QUERY: {query}, TOP_RECORDS: {top_records}")
    cleaned_query = query.strip()

    rt = RagTrace("tool", cleaned_query, top_records)
    cache_key = answer_cache.key("tool", cleaned_query, top_records, settings.ollama_model)
    try:
        cached = await answer_cache.get(cache_key)
        if cached is not None:
            try:
                answer = json.loads(cached)
                rt.cache = "hit"
                return answer
            except ValueError:
                pass

        with rt.stage("retrieve") as span:
            retrieved_chunks = await retry_async(
                partial(_retrieve, cleaned_query, top_records),
                **_RETRY,
                on_retry=partial(rt.retried, "retrieve"),
            )
            span.set_attribute("rag.chunks", len(retrieved_chunks))
        with rt.stage("generate"):
            llm_generated_data = await retry_async(
                partial(generate_sop_context, cleaned_query, retrieved_chunks),
                **_RETRY,
                on_retry=partial(rt.retried, "generate"),
            )

        if not llm_generated_data:
            return f"No SOP content found matching: {cleaned_query!r}"

        answer = llm_generated_data.model_dump()
        await answer_cache.set(cache_key, json.dumps(answer))
        return answer
    except Exception as e:
        rt.fail(e)
        logger.error("Retrieval failed for query %r: %s", cleaned_query, e)
        return f"Search failed: {e}"
    finally:
        rt.end()


async def stream_sop_answer(query: str, top_records: int = 5) -> AsyncIterator[str]:
    logger.info(f"STREAM QUERY: {query}, TOP_RECORDS: {top_records}")
    cleaned_query = query.strip()

    rt = RagTrace("stream", cleaned_query, top_records)
    cache_key = answer_cache.key("stream", cleaned_query, top_records, settings.ollama_model)
    try:
        cached = await answer_cache.get(cache_key)
        if cached is not None:
            rt.cache = "hit"
            for i in range(0, len(cached), _REPLAY_CHARS):
                rt.token()
                yield cached[i : i + _REPLAY_CHARS]
                await asyncio.sleep(0)
            return

        with rt.stage("retrieve") as span:
            retrieved_chunks = await retry_async(
                partial(_retrieve, cleaned_query, top_records),
                **_RETRY,
                on_retry=partial(rt.retried, "retrieve"),
            )
            span.set_attribute("rag.chunks", len(retrieved_chunks))

        parts: list[str] = []
        with rt.stage("generate"):
            async for token in retry_stream(
                lambda: stream_sop_context(cleaned_query, retrieved_chunks),
                **_RETRY,
                on_retry=partial(rt.retried, "generate"),
            ):
                rt.token()
                parts.append(token)
                yield token

        if parts:
            await answer_cache.set(cache_key, "".join(parts))
    except (asyncio.CancelledError, GeneratorExit):
        rt.outcome = "cancelled"
        raise
    except Exception as e:
        rt.fail(e)
        raise
    finally:
        rt.end()
