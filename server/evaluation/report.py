# Runs both eval suites (retrieval + generation) and renders a single
# self-contained HTML report: 4 headline stat tiles, then a per-question
# breakdown card for each suite. Open eval_report.html in a browser and
# screenshot it for the blog. Raw numbers are also saved to eval_report.json.
import asyncio
import html
import json
from datetime import date
from pathlib import Path

from evaluation.generation_evaluation import GROUND_TRUTH as GENERATION_GROUND_TRUTH
from evaluation.generation_evaluation import RAGGenerationEvaluation
from evaluation.retrieval_eval import GROUND_TRUTH as RETRIEVAL_GROUND_TRUTH
from evaluation.retrieval_eval import RetrivalMetrics

SERIES_A = "#2a78d6"  # blue - first metric of each pair
SERIES_B = "#eb6834"  # orange - second metric of each pair


async def run_retrieval() -> dict:
    metrics = RetrivalMetrics()
    tasks = [
        metrics.evaluate(question=item["question"], reference=item["reference"])
        for item in RETRIEVAL_GROUND_TRUTH
    ]
    await asyncio.gather(*tasks)
    return await metrics.get_results()


async def run_generation() -> dict:
    metrics = RAGGenerationEvaluation()
    tasks = [
        metrics.evaluate(question=item["question"], context=item["context"])
        for item in GENERATION_GROUND_TRUTH
    ]
    await asyncio.gather(*tasks)
    return await metrics.get_results()


def _avg(results: dict, key: str) -> float:
    values = [v[key] for v in results.values() if key in v]
    return sum(values) / len(values) if values else 0.0


def _bar_row(label: str, value: float, color: str) -> str:
    pct = max(0.0, min(1.0, value)) * 100
    return f"""
      <div class="bar-row">
        <span class="bar-label">{label}</span>
        <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>
        <span class="bar-value">{value:.0%}</span>
      </div>"""


def _question_block(question: str, rows: list[tuple[str, float, str]]) -> str:
    bars = "".join(_bar_row(label, value, color) for label, value, color in rows)
    return f"""
    <div class="question">
      <p class="question-text">{html.escape(question)}</p>
      {bars}
    </div>"""


def _stat_tile(label: str, value: float, color: str) -> str:
    return f"""
    <div class="tile">
      <span class="tile-dot" style="background:{color}"></span>
      <div class="tile-value">{value:.0%}</div>
      <div class="tile-label">{label}</div>
    </div>"""


def render_report(
    retrieval_results: dict, generation_results: dict, output_path: str
) -> None:
    tiles = "".join(
        [
            _stat_tile(
                "Context Precision",
                _avg(retrieval_results, "context_precision"),
                SERIES_A,
            ),
            _stat_tile(
                "Context Recall", _avg(retrieval_results, "context_recall"), SERIES_B
            ),
            _stat_tile(
                "Faithfulness",
                _avg(generation_results, "context_faithfulness"),
                SERIES_A,
            ),
            _stat_tile(
                "Answer Relevancy",
                _avg(generation_results, "context_relevance"),
                SERIES_B,
            ),
        ]
    )

    retrieval_blocks = "".join(
        _question_block(
            question,
            [
                ("Precision", scores["context_precision"], SERIES_A),
                ("Recall", scores["context_recall"], SERIES_B),
            ],
        )
        for question, scores in retrieval_results.items()
        if "context_precision" in scores and "context_recall" in scores
    )

    generation_blocks = "".join(
        _question_block(
            question,
            [
                ("Faithfulness", scores["context_faithfulness"], SERIES_A),
                ("Relevancy", scores["context_relevance"], SERIES_B),
            ],
        )
        for question, scores in generation_results.items()
        if "context_faithfulness" in scores and "context_relevance" in scores
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WMS SOP RAG — Evaluation Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: #f9f9f7; color: #0b0b0b;
    padding: 48px 24px;
  }}
  .page {{ max-width: 880px; margin: 0 auto; }}
  h1 {{ font-size: 26px; letter-spacing: -0.02em; }}
  .subtitle {{ color: #898781; font-size: 14px; margin-top: 6px; }}

  .tiles {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 28px 0; }}
  .tile {{
    background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10);
    border-radius: 14px; padding: 20px 18px 16px; position: relative;
  }}
  .tile-dot {{ position: absolute; top: 16px; right: 16px; width: 9px; height: 9px; border-radius: 50%; }}
  .tile-value {{ font-size: 34px; font-weight: 700; letter-spacing: -0.02em; }}
  .tile-label {{ font-size: 12.5px; color: #52514e; margin-top: 4px; }}

  .card {{
    background: #fcfcfb; border: 1px solid rgba(11,11,11,0.10);
    border-radius: 14px; padding: 24px; margin-bottom: 24px;
  }}
  .card h2 {{ font-size: 16px; margin-bottom: 2px; }}
  .card-sub {{ font-size: 12.5px; color: #898781; margin-bottom: 18px; }}

  .question {{ padding: 14px 0; border-top: 1px solid #e1e0d9; }}
  .question:first-of-type {{ border-top: none; }}
  .question-text {{ font-size: 13.5px; color: #0b0b0b; margin-bottom: 10px; font-weight: 500; }}

  .bar-row {{ display: flex; align-items: center; gap: 10px; margin: 5px 0; }}
  .bar-label {{ flex: 0 0 88px; font-size: 12px; color: #52514e; text-align: right; }}
  .bar-track {{ flex: 1; height: 10px; background: #f0efec; border-radius: 5px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 5px; }}
  .bar-value {{ flex: 0 0 44px; font-size: 12px; color: #52514e; font-variant-numeric: tabular-nums; }}

  .legend {{ display: flex; gap: 16px; margin-bottom: 4px; }}
  .legend span {{ font-size: 12px; color: #52514e; display: flex; align-items: center; gap: 6px; }}
  .legend i {{ width: 10px; height: 10px; border-radius: 3px; display: inline-block; }}

  .footer {{ font-size: 12px; color: #898781; margin-top: 8px; }}
</style>
</head>
<body>
<div class="page">
  <h1>WMS SOP RAG — Evaluation Report</h1>
  <p class="subtitle">Retrieval and generation measured independently with ragas ·
    {len(retrieval_results)} retrieval questions · {len(generation_results)} generation questions · {date.today():%d %b %Y}</p>

  <div class="tiles">{tiles}</div>

  <div class="card">
    <h2>Retrieval — per question</h2>
    <p class="card-sub">Hybrid search (pgvector + full-text) scored against hand-written references from the SOP PDF</p>
    <div class="legend">
      <span><i style="background:{SERIES_A}"></i>Context Precision</span>
      <span><i style="background:{SERIES_B}"></i>Context Recall</span>
    </div>
    {retrieval_blocks}
  </div>

  <div class="card">
    <h2>Generation — per question</h2>
    <p class="card-sub">Fixed known-correct contexts (with noise chunks), so scores isolate the generator</p>
    <div class="legend">
      <span><i style="background:{SERIES_A}"></i>Faithfulness</span>
      <span><i style="background:{SERIES_B}"></i>Answer Relevancy</span>
    </div>
    {generation_blocks}
  </div>

  <p class="footer">Judge model: gpt-4.1-mini · Generator: gemma4:31b-cloud via Ollama · Embeddings: text-embedding-3-small</p>
</div>
</body>
</html>"""

    Path(output_path).write_text(page, encoding="utf-8")


async def main() -> None:
    retrieval_results, generation_results = await asyncio.gather(
        run_retrieval(), run_generation()
    )

    Path("eval_report.json").write_text(
        json.dumps(
            {"retrieval": retrieval_results, "generation": generation_results},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    render_report(retrieval_results, generation_results, "eval_report.html")
    print("Report saved to eval_report.html (raw numbers in eval_report.json)")
    print("Open it in a browser and screenshot for the blog.")


if __name__ == "__main__":
    asyncio.run(main())
