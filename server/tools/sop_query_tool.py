import logging
from typing import Any

from config.settings import settings
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from tools.rag_generator import generate_sop_context

logger = logging.getLogger(__name__)

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
        A list of matching SOP chunks containing:
        - relevance score
        - chunk text
        - page information
        - generated title
        - generated questions
        - source metadata
    """
    logger.info(f"QUERY: {query}, TOP_RECORDS: {top_records}")
    cleaned_query = query.strip()

    retriever = settings.index.as_retriever(
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
        similarity_top_k=top_records,
    )

    try:
        vector_retrieved_data = await retriever.aretrieve(cleaned_query)

        retrieved_chunks = [
            {
                "text": item.node.get_content(),
                "pages": item.node.metadata.get("covered_pages"),
                "title": item.node.metadata.get("document_title"),
                "score": item.score,
            }
            for item in vector_retrieved_data
        ]

        llm_generated_data = await generate_sop_context(cleaned_query, retrieved_chunks)
    except Exception as e:
        logger.error("Retrieval failed for query %r: %s", cleaned_query, e)
        return f"Search failed: {e}"

    if not llm_generated_data:
        return f"No SOP content found matching: {cleaned_query!r}"

    return llm_generated_data.model_dump()
