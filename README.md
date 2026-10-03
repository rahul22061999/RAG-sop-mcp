# WMS SOP Assistant

RAG chat and [MCP](https://modelcontextprotocol.io) tool that answers warehouse SOP questions from a PDF, with page citations.

![Chat UI with SOP question suggestions](docs/chat-welcome.png)
![Streamed answer with page sources](docs/chat-answer.png)

## Stack

| Part | Tool |
|---|---|
| Parsing | Docling + VLM (`gemma4:31b-cloud` via Ollama), page image to Markdown |
| Chunking | LlamaIndex: page + next page context, 1200 chars, 200 overlap, LLM title + 3 questions per chunk |
| Embeddings | `embeddinggemma:300m` (768-d) |
| Store | Postgres + pgvector, hybrid search (vector + full-text) |
| Cache | Redis, 300 s TTL |
| API | FastAPI WebSocket (UI), FastMCP + Unkey auth (agents) |
| UI | Next.js, assistant-ui |
| Observability | OpenTelemetry, Prometheus, Jaeger |
| Eval | ragas |

## Flow

```
PDF -> VLM -> Markdown -> chunk + tag -> embed -> pgvector

question -> Redis --hit--> answer
                 \-miss--> hybrid search (top 5) -> LLM (grounded, cited) -> cache 300 s
```

UI to server: one WebSocket per question. Client sends `{"query"}`, server streams `{"type":"token"}` messages, then `{"type":"done"}` or `{"type":"error"}`. Disconnect cancels the search and the LLM call.

## Run

```bash
docker run -d --name pgvector-db -p 5432:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=mydb pgvector/pgvector:pg16
docker run -d --name wms-redis -p 6379:6379 redis:7-alpine      # optional
ollama pull embeddinggemma:300m
cd server && python -m pipeline.main "data/<sop>.pdf"            # parse, chunk, embed, store
uvicorn api.main:app --port 8000                                 # chat API
cd ../frontend/sop-chatbot && npm install && npm run dev         # http://localhost:3000
```

MCP server: `wms-sop-mcp start` (port 8001). Config in `.env` (see `server/config/settings.py`).

## Reliability and latency

- **Cache:** Redis, 300 s TTL, key ignores case and spacing. Only complete answers are cached. Redis down means a cache miss, not an error.
- **Retries:** transient errors (network, timeout, 429/5xx) retried 3 times with exponential backoff and full jitter. Streams retry only before the first token.
- **Tracing:** every request is a trace: `rag.request` > `rag.retrieve`, `rag.generate`.
- **Percentiles:** `cd server && python latency_report.py http://localhost:8000/metrics`

Local run, 8 distinct questions then 40 repeats:

| | n | p50 | p90 | p99 |
|---|---|---|---|---|
| Full answer, miss | 8 | 2.0 s | 8.8 s | 11.7 s |
| Full answer, hit | 40 | 3 ms | 5 ms | 8 ms |
| First token, miss | 8 | 0.4 s | 5.6 s | 7.8 s |
| First token, hit | 40 | 2 ms | 5 ms | 5 ms |

Miss sample is small, so its p99 is noisy. Jaeger: `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318`.

## Evaluation

`python -m evaluation.report` (ragas, `gpt-4.1-mini` judge), retrieval and generation scored separately. August run, before the VLM and Ollama embedding changes:

| Metric | Result |
|---|---|
| Context precision | 0.88 |
| Context recall | 1.00 |
| Faithfulness | 0.97 |
| Answer relevancy | 0.66 (0.74 excluding the out-of-context trap question) |

## Design decisions

- **VLM parsing:** SOPs have screenshots and tables that plain OCR loses.
- **Cross-page chunks:** procedures that span a page break stay whole.
- **Title and question tags:** improve recall on reworded questions.
- **Hybrid search:** meaning plus exact terms.
- **Grounded or refused:** answers only from retrieved text, says so when not covered.
- **Cancellable streaming:** no compute spent on abandoned answers.
