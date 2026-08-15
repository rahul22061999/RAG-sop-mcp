from __future__ import annotations

import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    VlmConvertOptions,
    VlmPipelineOptions,
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


logger = logging.getLogger("docling_vlm_ingest")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


DeviceName = Literal["auto", "cpu", "mps", "cuda", "xpu"]


class IngestedDocument(BaseModel):
    source_file: str
    file_name: str
    parser: str = "docling"
    pipeline: str = "vlm"
    vlm_preset: str
    device: str
    num_threads: int
    status: str
    elapsed_seconds: float
    ingested_at_utc: str = Field(
        default_factory=lambda: datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
    )
    markdown_path: str | None = None
    json_path: str | None = None
    markdown: str | None = None
    document: dict[str, Any] | None = None
    error: str | None = None


class IngestDocument:
    """
    Production-style Docling VLM ingestion pipeline.

    Use this when you want:
    - local or accelerated VLM parsing
    - markdown output for RAG
    - JSON metadata for audit/layout/table/image information
    """

    def __init__(
        self,
        output_dir: str | Path = "ingested_docs",
        device: DeviceName = "auto",
        num_threads: int = 8,
        vlm_preset: str = "granite_docling",
        enable_profiling: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.markdown_dir = self.output_dir / "markdown"
        self.json_dir = self.output_dir / "json"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)

        self.device = device
        self.num_threads = num_threads
        self.vlm_preset = vlm_preset
        self.enable_profiling = enable_profiling

        if enable_profiling:
            settings.debug.profile_pipeline_timings = True

        self.converter = self._build_converter()

    def _to_accelerator_device(self) -> AcceleratorDevice:
        device_map = {
            "auto": AcceleratorDevice.AUTO,
            "cpu": AcceleratorDevice.CPU,
            "mps": AcceleratorDevice.MPS,
            "cuda": AcceleratorDevice.CUDA,
            "xpu": AcceleratorDevice.XPU,
        }
        return device_map[self.device]

    def _build_converter(self) -> DocumentConverter:
        """
        Build Docling VLM converter.

        VlmConvertOptions.from_preset("granite_docling") uses Docling's
        recommended GraniteDocling VLM preset. Docling docs show this as
        the recommended explicit VLM setup.
        """
        accelerator_options = AcceleratorOptions(
            num_threads=self.num_threads,
            device=self._to_accelerator_device(),
        )

        vlm_options = VlmConvertOptions.from_preset(self.vlm_preset)

        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_options,
        )

        if hasattr(pipeline_options, "accelerator_options"):
            pipeline_options.accelerator_options = accelerator_options

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                )
            }
        )

        return converter

    def ingest_one(
        self,
        file_path: str | Path,
        include_raw_document_in_json: bool = True,
    ) -> IngestedDocument:
        input_path = Path(file_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Document not found: {input_path}")

        if not input_path.is_file():
            raise ValueError(f"Path is not a file: {input_path}")

        started = time.perf_counter()
        logger.info("Starting ingestion: %s", input_path)

        markdown_path = self.markdown_dir / f"{input_path.stem}.md"
        json_path = self.json_dir / f"{input_path.stem}.json"

        try:
            result = self.converter.convert(str(input_path))
            elapsed = round(time.perf_counter() - started, 3)

            status = getattr(result, "status", None)

            if status and status != ConversionStatus.SUCCESS:
                logger.warning("Conversion finished with status: %s", status)

            markdown = result.document.export_to_markdown()
            doc_dict = result.document.export_to_dict()

            markdown_path.write_text(markdown, encoding="utf-8")

            output = IngestedDocument(
                source_file=str(input_path),
                file_name=input_path.name,
                vlm_preset=self.vlm_preset,
                device=self.device,
                num_threads=self.num_threads,
                status=str(status or "unknown"),
                elapsed_seconds=elapsed,
                markdown_path=str(markdown_path),
                json_path=str(json_path),
                markdown=markdown,
                document=doc_dict if include_raw_document_in_json else None,
            )

            json_path.write_text(
                output.model_dump_json(indent=2, exclude_none=True),
                encoding="utf-8",
            )

            logger.info(
                "Finished ingestion: %s in %.3fs",
                input_path.name,
                elapsed,
            )

            return output

        except Exception as exc:
            elapsed = round(time.perf_counter() - started, 3)
            logger.exception("Failed ingestion: %s", input_path)

            output = IngestedDocument(
                source_file=str(input_path),
                file_name=input_path.name,
                vlm_preset=self.vlm_preset,
                device=self.device,
                num_threads=self.num_threads,
                status="failed",
                elapsed_seconds=elapsed,
                json_path=str(json_path),
                error=str(exc),
            )

            json_path.write_text(
                output.model_dump_json(indent=2, exclude_none=True),
                encoding="utf-8",
            )

            return output

    def ingest_many(
        self,
        input_dir: str | Path,
        patterns: tuple[str, ...] = ("*.pdf",),
        include_raw_document_in_json: bool = True,
    ) -> list[IngestedDocument]:
        input_dir = Path(input_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        files: list[Path] = []

        for pattern in patterns:
            files.extend(input_dir.glob(pattern))

        files = sorted(set(files))

        logger.info("Found %s files", len(files))

        results: list[IngestedDocument] = []

        for file_path in files:
            result = self.ingest_one(
                file_path=file_path,
                include_raw_document_in_json=include_raw_document_in_json,
            )
            results.append(result)

        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                [result.model_dump(exclude_none=True) for result in results],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return results