import logging
from collections.abc import AsyncIterator
from typing import Any

from config.settings import settings
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from tools.rag_generator import generate_sop_context, stream_sop_context

logger = logging.getLogger(__name__)


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

    try:
        retrieved_chunks = await _retrieve(cleaned_query, top_records)
        llm_generated_data = await generate_sop_context(cleaned_query, retrieved_chunks)
    except Exception as e:
        logger.error("Retrieval failed for query %r: %s", cleaned_query, e)
        return f"Search failed: {e}"

    if not llm_generated_data:
        return f"No SOP content found matching: {cleaned_query!r}"

    return llm_generated_data.model_dump()


async def stream_sop_answer(query: str, top_records: int = 5) -> AsyncIterator[str]:
    """Streaming variant for the websocket: yields answer tokens.

    Errors propagate to the caller so it can report them as an error message
    instead of mixing them into the answer text.
    """
    logger.info(f"STREAM QUERY: {query}, TOP_RECORDS: {top_records}")
    cleaned_query = query.strip()

    retrieved_chunks = await _retrieve(cleaned_query, top_records)
    async for token in stream_sop_context(cleaned_query, retrieved_chunks):
        yield token
