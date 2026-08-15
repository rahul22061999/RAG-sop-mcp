from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    TableStructureOptions,
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

logger = logging.getLogger(__name__)


class DocumentIngestor:
    """
    Ingest documents with Docling and export markdown/json.
    Uses Docling's threaded PDF pipeline for page-stage parallelism.
    """

    def __init__(
        self,
        device: str = "auto",
        num_threads: int = 8,
        enable_ocr: bool = True,
        enable_tables: bool = True,
        enable_picture_description: bool = True,
        include_raw_document: bool = True,
    ) -> None:
        self.device = device
        self.num_threads = num_threads
        self.enable_ocr = enable_ocr
        self.enable_tables = enable_tables
        self.enable_picture_description = enable_picture_description
        self.include_raw_document = include_raw_document

        # Docling docs mention OMP_NUM_THREADS for CPU thread limiting.
        # Keep it aligned with AcceleratorOptions.
        os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))

        self.converter = self._build_converter()

    def _build_converter(self) -> DocumentConverter:
        device_map = {
            "auto": AcceleratorDevice.AUTO,
            "cpu": AcceleratorDevice.CPU,
            "mps": AcceleratorDevice.MPS,
            "cuda": AcceleratorDevice.CUDA,
            "xpu": AcceleratorDevice.XPU,
        }

        if self.device not in device_map:
            raise ValueError(
                f"Invalid device={self.device}. Use one of: {list(device_map.keys())}"
            )

        accelerator_options = AcceleratorOptions(
            num_threads=self.num_threads,
            device=device_map[self.device],
        )

        # Out-of-box threaded PDF pipeline.
        pipeline_options = ThreadedPdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options

        pipeline_options.do_ocr = self.enable_ocr
        pipeline_options.do_table_structure = self.enable_tables
        pipeline_options.do_picture_description = self.enable_picture_description

        if self.enable_tables:
            pipeline_options.table_structure_options = TableStructureOptions(
                do_cell_matching=True
            )

        logger.info(
            "Docling config: device=%s threads=%s ocr=%s tables=%s picture_description=%s raw=%s",
            self.device,
            self.num_threads,
            self.enable_ocr,
            self.enable_tables,
            self.enable_picture_description,
            self.include_raw_document,
        )

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

    def ingest_one(
        self,
        file_path: str | Path,
        output_json: str | Path,
        output_md: str | Path,
    ) -> dict[str, Any]:
        input_path = Path(file_path)
        output_json_path = Path(output_json)
        output_md_path = Path(output_md)

        if not input_path.exists():
            raise FileNotFoundError(f"Document not found: {input_path}")

        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_md_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Starting Docling ingestion: %s", input_path)

        started = time.perf_counter()

        result = self.converter.convert(str(input_path))

        elapsed_seconds = round(time.perf_counter() - started, 3)

        logger.info(
            "Docling conversion completed: file=%s elapsed_seconds=%s",
            input_path.name,
            elapsed_seconds,
        )

        markdown = result.document.export_to_markdown()
        doc_dict = result.document.export_to_dict() if self.include_raw_document else None

        output_md_path.write_text(markdown, encoding="utf-8")

        output = {
            "source_file": str(input_path),
            "file_name": input_path.name,
            "parser": "docling",
            "device": self.device,
            "num_threads": self.num_threads,
            "elapsed_seconds": elapsed_seconds,
            "markdown_file": str(output_md_path),
            "settings": {
                "enable_ocr": self.enable_ocr,
                "enable_tables": self.enable_tables,
                "enable_picture_description": self.enable_picture_description,
                "include_raw_document": self.include_raw_document,
                "pipeline": "threaded_pdf",
            },
            "markdown": markdown,
            "document": doc_dict,
        }

        output_json_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(
            "Saved Docling outputs: json=%s markdown=%s",
            output_json_path,
            output_md_path,
        )

        return output