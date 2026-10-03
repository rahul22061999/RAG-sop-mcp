import asyncio
import sys
from pathlib import Path

from pipeline.chunk_and_embed import DocumentChunkEmbedPipeline
from pipeline.ingestion import DocumentIngestor

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


async def run_pipeline(pdf_path: str | Path, output_dir: str | Path = OUTPUT_DIR) -> None:
    pdf, out = Path(pdf_path), Path(output_dir)
    json_path = out / f"{pdf.stem}.json"


    await asyncio.to_thread(
        DocumentIngestor().ingest_one, pdf, json_path, out / f"{pdf.stem}.md"
    )
    await DocumentChunkEmbedPipeline(
        json_input_path=json_path,
        document_output_path=out / f"{pdf.stem}_documents.json",
        node_output_path=out / f"{pdf.stem}_nodes.json",
    ).arun()


if __name__ == "__main__":
    asyncio.run(run_pipeline(sys.argv[1]))
