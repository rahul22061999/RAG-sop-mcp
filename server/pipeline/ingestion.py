from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from pipeline.prompt import PAGE_TO_MARKDOWN_PROMPT
from config.settings import Settings

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options_vlm_model import ResponseFormat
from docling.datamodel.stage_model_specs import VlmModelSpec
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.models.inference_engines.vlm.base import VlmEngineType

from docling.datamodel.pipeline_options import (
    VlmConvertOptions,
    VlmPipelineOptions,
)
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.document_converter import DocumentConverter, PdfFormatOption

logger = logging.getLogger(__name__)


class DocumentIngestor:

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config if config is not None else Settings()
        self.converter = self._build_converter()

    def _build_converter(self) -> DocumentConverter:

        model_name = self.config.ollama_model

        engine_options = ApiVlmEngineOptions(
            engine_type=VlmEngineType.API_OLLAMA,
            url=f"{self.config.ollama_base_url.rstrip('/')}/v1/chat/completions",
            params={"model": model_name},
            timeout=self.config.ollama_request_timeout,
            concurrency=2,
        )

        vlm_options = VlmConvertOptions(
            model_spec=VlmModelSpec(
                name=model_name,
                default_repo_id=model_name,
                prompt=PAGE_TO_MARKDOWN_PROMPT,
                response_format=ResponseFormat.MARKDOWN,
                supported_engines={VlmEngineType.API_OLLAMA},
            ),
            engine_options=engine_options,
            scale=2.0,
        )

        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_options,
            enable_remote_services=True,
        )

        logger.info(
            "Configured VLM model=%s endpoint=%s",
            model_name,
            self.config.ollama_base_url,
        )

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                )
            }
        )

    def ingest_one(
        self,
        input_path: str | Path,
        json_path: str | Path,
        md_path: str | Path,
    ) -> dict[str, Any]:
        input_path = Path(input_path)
        json_path = Path(json_path)
        md_path = Path(md_path)

        if not input_path.is_file():
            raise FileNotFoundError(f"PDF not found: {input_path}")
        if json_path.suffix.lower() != ".json":
            raise ValueError("output_json must be a .json file path")
        if md_path.suffix.lower() != ".md":
            raise ValueError("output_md must be a .md file path")

        logger.info("Starting VLM conversion: %s", input_path)
        started = time.perf_counter()

        result = self.converter.convert(str(input_path))

        if result.status != ConversionStatus.SUCCESS:
            raise RuntimeError(f"Docling conversion did not compelte: {result.status}")

        markdown = result.document.export_to_markdown()
        if not markdown.strip():
            raise RuntimeError(
                f"Docling returned no text for {input_path.name}; check the VLM endpoint/model"
            )
        elapsed_time = round(time.perf_counter() - started, 3)

        output = {
            "source_file": str(input_path),
            "file_name": input_path.name,
            "parser": "docling",
            "parser": "docling",
            "model": self.config.ollama_model,
            "elapsed_seconds": elapsed_time,
            "markdown_file": str(md_path),
            "settings": {
                "pipeline": "vlm",
                "response_format": "markdown",
                "image_scale": 2.0,
            },
            "markdown": markdown,
            "document": result.document.export_to_dict(),
        }

        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            md_path.write_text(markdown, encoding="utf-8")
            json_path.write_text(
                json.dumps(output, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            logger.exception("Could not write Files to  path error%e", e)

        logger.info(
            "Saved VLM outputs: markdown=%s json=%s, elapsed=%s",
            md_path,
            json_path,
            elapsed_time,
        )

        return output
