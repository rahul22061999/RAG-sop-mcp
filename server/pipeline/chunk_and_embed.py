from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from config.settings import Settings
from docling_core.types import DoclingDocument
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.extractors import (
    QuestionsAnsweredExtractor,
    TitleExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy.engine import URL

logger = logging.getLogger(__name__)


class DocumentChunkEmbedPipeline:

    def __init__(
        self,
        json_input_path: str | Path,
        document_output_path: str | Path,
        node_output_path: str | Path,
        module: str = "DTAC",
        system: str = "WMS",
        doc_type: str = "sop",
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        next_page_context_chars: int = 2000,
        title_nodes: int = 5,
        questions_per_chunk: int = 3,
        config: Settings | None = None
    ) -> None:
        self.config = config if config else Settings()
        self.json_path = Path(json_input_path)
        self.document_output_path = Path(document_output_path)
        self.node_output_path = Path(node_output_path)

        self.module = module
        self.system = system
        self.doc_type = doc_type
        self.next_page_context_chars = next_page_context_chars

        
        self.llm= Ollama(
            base_url=self.config.ollama_base_url, 
            model=self.config.ollama_model,
            request_timeout=self.config.ollama_request_timeout
        )

        self.embed_model = OllamaEmbedding(
            base_url=self.config.ollama_embedding_url, 
            model_name=self.config.ollama_embedding_model,
            request_timeout=self.config.ollama_request_timeout
        )

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

        self.node_pipeline = IngestionPipeline(
            transformations=[
                self.splitter,
                self.title_extractor,
                self.questions_extractor,
            ]
        )

    def load_docling_json(self) -> dict[str, Any]:
        if not self.json_path.exists():
            raise FileNotFoundError(f"Docling JSON not found: {self.json_path}")

        logger.info("Loading Docling JSON: %s", self.json_path)

        return json.loads(self.json_path.read_text(encoding="utf-8"))

    def build_page_documents(self) -> list[Document]:

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

    async def split_and_enrich_nodes(
        self,
        chunking_documents: list[Document],
    ) -> list[BaseNode]:
        logger.info(
            "Splitting and enriching %s chunking Documents",
            len(chunking_documents),
        )

        nodes = await self.node_pipeline.arun(
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

    def save_nodes_to_pg(
        self,
        nodes: list[BaseNode],
    ) -> None:
        logger.info(
            "Embedding and saving %s nodes to Postgres table '%s.%s'",
            len(nodes),
            self.config.pg_schema_name,
            self.config.pg_table_name,
        )

        vector_store = self.config.vector_store

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

    async def arun(self) -> None:
        page_documents = self.build_page_documents()
        self.save_documents_preview(page_documents)

        chunking_documents = self.build_cross_page_chunking_documents(
            page_documents
        )
        nodes = await self.split_and_enrich_nodes(chunking_documents)
        self.save_nodes_preview(nodes)

        self.save_nodes_to_pg(nodes)

        logger.info("Pipeline finished. Data saved to Postgres pgvector.")

    def run(self) -> None:
        asyncio.run(self.arun())
