from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from docling_core.types import DoclingDocument
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.extractors import (
    QuestionsAnsweredExtractor,
    TitleExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy.engine import URL

logger = logging.getLogger(__name__)


class SOPLLMCleaner(TransformComponent):
    """Clean OCR-extracted SOP text before chunking."""

    llm: Any

    def __init__(self, llm: Any) -> None:
        super().__init__(llm=llm)

    def __call__(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> list[BaseNode]:
        cleaned_nodes: list[BaseNode] = []
        total = len(nodes)

        for index, node in enumerate(nodes, start=1):
            logger.info("Cleaning page %s/%s", index, total)

            original_text = node.text or ""

            if not original_text.strip():
                cleaned_nodes.append(node)
                continue

            cleaned_text = self.clean_text(original_text)
            node.set_content(cleaned_text)
            cleaned_nodes.append(node)

        return cleaned_nodes

    def clean_text(self, text: str) -> str:
        prompt = f"""You are cleaning OCR-extracted warehouse SOP text.

Fix:
- obvious OCR mistakes
- broken line breaks
- broken spacing
- obvious button-label OCR errors when the intended label is clear
- generic image-description phrasing that adds no useful information

Important:
- Do not summarize.
- Do not add new instructions.
- Do not remove procedural steps.
- Preserve headings, numbered steps, bullets, warnings, and tables.
- Preserve WMS-specific names unless they are obvious OCR mistakes.
- Preserve all page-continuation information.
- If a button label is clearly wrong because of OCR, correct it.
- Example: "OK & Net" should become "OK & Next".
- Return only the cleaned text.

Text:
{text}"""

        return str(self.llm.complete(prompt)).strip()


class DocumentIngestChunkEmbedPipeline:
    """
    Pipeline:

    Docling JSON
    -> one Document per page
    -> clean each page once
    -> build current-page + next-page chunking windows
    -> split windows into overlapping nodes
    -> add title and question metadata
    -> embed nodes
    -> save nodes into Postgres (pgvector)
    """

    def __init__(
        self,
        json_input_path: str | Path,
        document_output_path: str | Path,
        node_output_path: str | Path,
        module: str = "DTAC",
        system: str = "WMS",
        doc_type: str = "sop",
        llm_model: str = "gpt-4.1-mini",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        next_page_context_chars: int = 2000,
        title_nodes: int = 5,
        questions_per_chunk: int = 3,
        openai_api_key: str | None = None,
        pg_host: str | None = None,
        pg_port: int | None = None,
        pg_database: str | None = None,
        pg_user: str | None = None,
        pg_password: str | None = None,
        pg_table_name: str | None = None,
        pg_schema_name: str = "public",
        pg_embed_dim: int = 1536,
    ) -> None:
        self.json_path = Path(json_input_path)
        self.document_output_path = Path(document_output_path)
        self.node_output_path = Path(node_output_path)

        self.module = module
        self.system = system
        self.doc_type = doc_type
        self.next_page_context_chars = next_page_context_chars

        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        self.pg_host = pg_host or os.getenv("PG_HOST")
        self.pg_port = pg_port or int(os.getenv("PG_PORT", "5432"))
        self.pg_database = pg_database or os.getenv("PG_DATABASE")
        self.pg_user = pg_user or os.getenv("PG_USER")
        self.pg_password = pg_password or os.getenv("PG_PASSWORD")
        self.pg_table_name = pg_table_name or os.getenv("PG_TABLE_NAME") or "sop_chunks"
        self.pg_schema_name = pg_schema_name
        self.pg_embed_dim = pg_embed_dim
        self.pg_ssl_mode = os.getenv("PG_SSL_MODE", "disable")

        self._validate_configuration()

        self.llm = OpenAI(
            model=llm_model,
            temperature=0,
            api_key=self.openai_api_key,
        )

        self.embed_model = OpenAIEmbedding(
            model=embedding_model,
            api_key=self.openai_api_key,
        )

        self.cleaner = SOPLLMCleaner(llm=self.llm)

        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.title_extractor = TitleExtractor(
            llm=self.llm,
            nodes=title_nodes,
        )

        self.questions_extractor = QuestionsAnsweredExtractor(
            llm=self.llm,
            questions=questions_per_chunk,
        )

        # Clean original page Documents once.
        self.cleaning_pipeline = IngestionPipeline(
            transformations=[
                self.cleaner,
            ]
        )

        # Split temporary cross-page Documents and enrich resulting nodes.
        self.node_pipeline = IngestionPipeline(
            transformations=[
                self.splitter,
                self.title_extractor,
                self.questions_extractor,
            ]
        )

    def _validate_configuration(self) -> None:
        if not self.openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY")

        if not self.pg_host:
            raise ValueError("Missing PG_HOST")

        if not self.pg_database:
            raise ValueError("Missing PG_DATABASE")

        if not self.pg_user:
            raise ValueError("Missing PG_USER")

        if not self.pg_password:
            raise ValueError("Missing PG_PASSWORD")

        if self.next_page_context_chars < 0:
            raise ValueError("next_page_context_chars cannot be negative")

    def load_docling_json(self) -> dict[str, Any]:
        if not self.json_path.exists():
            raise FileNotFoundError(f"Docling JSON not found: {self.json_path}")

        logger.info("Loading Docling JSON: %s", self.json_path)

        return json.loads(self.json_path.read_text(encoding="utf-8"))

    def build_page_documents(self) -> list[Document]:
        """
        Create exactly one LlamaIndex Document per Docling page.
        """

        data = self.load_docling_json()
        docling_document = DoclingDocument.model_validate(data["document"])

        page_documents: list[Document] = []
        total_pages = len(docling_document.pages)

        logger.info(
            "Building one Document per page. total_pages=%s",
            total_pages,
        )

        for page_key in sorted(
            docling_document.pages.keys(),
            key=int,
        ):
            page_number = int(page_key)

            page_markdown = docling_document.export_to_markdown(
                page_no=page_number,
                image_placeholder="",
                escape_html=False,
                traverse_pictures=True,
                enable_chart_tables=True,
            )

            if not page_markdown.strip():
                logger.info(
                    "Skipping empty page %s",
                    page_number,
                )
                continue

            page_document = Document(
                text=page_markdown.strip(),
                metadata={
                    "file_name": data.get("file_name"),
                    "source_file": data.get("source_file"),
                    "page_number": page_number,
                    "total_pages": total_pages,
                    "module": self.module,
                    "system": self.system,
                    "doc_type": self.doc_type,
                    "parser": "docling",
                    "content_type": "page_markdown",
                },
                excluded_llm_metadata_keys=["source_file"],
                excluded_embed_metadata_keys=["source_file"],
            )

            page_documents.append(page_document)

        logger.info(
            "Created %s page Documents",
            len(page_documents),
        )

        return page_documents

    def clean_page_documents(
        self,
        page_documents: list[Document],
    ) -> list[Document]:
        logger.info(
            "Cleaning %s page Documents",
            len(page_documents),
        )

        cleaned_documents = self.cleaning_pipeline.run(
            documents=page_documents,
            show_progress=True,
        )

        return list(cleaned_documents)

    def build_cross_page_chunking_documents(
        self,
        page_documents: list[Document],
    ) -> list[Document]:
        chunking_documents: list[Document] = []

        for index, current_document in enumerate(page_documents):
            current_page = int(current_document.metadata["page_number"])

            next_document = (
                page_documents[index + 1] if index + 1 < len(page_documents) else None
            )

            covered_pages = [current_page]
            next_page: int | None = None

            combined_text = f"[PAGE {current_page}]\n{current_document.text}"

            if next_document is not None:
                next_page = int(next_document.metadata["page_number"])
                covered_pages.append(next_page)

                next_page_text = next_document.text[: self.next_page_context_chars]

                combined_text += (
                    f"\n\n[CONTINUATION FROM PAGE {next_page}]\n{next_page_text}"
                )

            chunking_document = Document(
                text=combined_text,
                metadata={
                    **current_document.metadata,
                    "primary_page_number": current_page,
                    "next_page_number": next_page,
                    "covered_pages": covered_pages,
                    "cross_page_context": next_document is not None,
                    "content_type": "cross_page_chunking_window",
                },
                excluded_llm_metadata_keys=["source_file"],
                excluded_embed_metadata_keys=["source_file"],
            )

            chunking_documents.append(chunking_document)

        logger.info(
            "Created %s cross-page chunking Documents",
            len(chunking_documents),
        )

        return chunking_documents

    def split_and_enrich_nodes(
        self,
        chunking_documents: list[Document],
    ) -> list[BaseNode]:

        logger.info(
            "Splitting and enriching %s chunking Documents",
            len(chunking_documents),
        )

        nodes = self.node_pipeline.run(
            documents=chunking_documents,
            show_progress=True,
        )

        logger.info(
            "Produced %s enriched nodes",
            len(nodes),
        )

        return list(nodes)

    def save_documents_preview(
        self,
        documents: Sequence[BaseNode],
    ) -> None:
        self.document_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = [
            {
                "document_index": index,
                "document_id": document.node_id,
                "text": document.text,
                "metadata": document.metadata,
            }
            for index, document in enumerate(documents)
        ]

        self.document_output_path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        logger.info(
            "Saved page Document preview: %s",
            self.document_output_path,
        )

    def save_nodes_preview(
        self,
        nodes: Sequence[BaseNode],
    ) -> None:
        self.node_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = [
            {
                "node_index": index,
                "node_id": node.node_id,
                "text": node.text,
                "metadata": node.metadata,
                "primary_page_number": node.metadata.get("primary_page_number"),
                "next_page_number": node.metadata.get("next_page_number"),
                "covered_pages": node.metadata.get("covered_pages"),
                "document_title": node.metadata.get("document_title"),
                "questions_this_excerpt_can_answer": (
                    node.metadata.get("questions_this_excerpt_can_answer")
                ),
            }
            for index, node in enumerate(nodes)
        ]

        self.node_output_path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        logger.info(
            "Saved enriched-node preview: %s",
            self.node_output_path,
        )

    def create_pg_vector_store(
        self,
    ) -> PGVectorStore:
        base_url = URL.create(
            drivername="postgresql",
            username=self.pg_user,
            password=self.pg_password,
            host=self.pg_host,
            port=self.pg_port,
            database=self.pg_database,
        )
        connection_string = base_url.set(
            drivername="postgresql+psycopg2",
            query={"sslmode": self.pg_ssl_mode},
        ).render_as_string(hide_password=False)
        async_connection_string = base_url.set(
            drivername="postgresql+asyncpg",
            query=({} if self.pg_ssl_mode == "disable" else {"ssl": self.pg_ssl_mode}),
        ).render_as_string(hide_password=False)

        return PGVectorStore(
            connection_string=connection_string,
            async_connection_string=async_connection_string,
            table_name=self.pg_table_name,
            schema_name=self.pg_schema_name,
            embed_dim=self.pg_embed_dim,
            hybrid_search=True,
            text_search_config="english",
        )

    def save_nodes_to_pg(
        self,
        nodes: list[BaseNode],
    ) -> None:
        logger.info(
            "Embedding and saving %s nodes to Postgres table '%s.%s'",
            len(nodes),
            self.pg_schema_name,
            self.pg_table_name,
        )

        vector_store = self.create_pg_vector_store()

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
        )

        VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=True,
        )

        logger.info("Postgres pgvector indexing completed")

    def run(self) -> None:
        page_documents = self.build_page_documents()
        cleaned_page_documents = self.clean_page_documents(page_documents)
        self.save_documents_preview(cleaned_page_documents)

        chunking_documents = self.build_cross_page_chunking_documents(
            cleaned_page_documents
        )
        nodes = self.split_and_enrich_nodes(chunking_documents)
        self.save_nodes_preview(nodes)

        self.save_nodes_to_pg(nodes)

        logger.info("Pipeline finished. Data saved to Postgres pgvector.")
