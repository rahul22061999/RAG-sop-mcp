# wms-sop-mcp

An [MCP](https://modelcontextprotocol.io) server that answers questions about
warehouse Standard Operating Procedures using hybrid retrieval over a
Postgres/pgvector store. Built to sit behind an agent (e.g. a LangGraph deep
agent) as the "how do we do X per the SOP" tool, alongside a sibling
`wms_sql_mcp` server that answers "what's actually happening in the
warehouse right now" from live operational data.

## How it works

```
question --> hybrid retriever (pgvector cosine + Postgres full-text)
         --> top-k SOP chunks, each tagged with its source page(s)
         --> Ollama (gemma-family cloud model) grounds an answer in
             only the retrieved chunks, cites pages, scores confidence
         --> { answer, citations, confidence }
```

- **Retrieval**: `llama-index-vector-stores-postgres`, `HYBRID` query mode
  (`VectorStoreQueryMode.HYBRID`) — combines pgvector cosine similarity with
  Postgres `tsvector` full-text search, so both semantic and keyword-exact
  queries work.
- **Generation**: deliberately kept separate from retrieval so faithfulness
  and relevancy can be evaluated in isolation (see `server/evaluation/`).
  Runs against an Ollama endpoint, including Ollama's `-cloud` models.
- **Auth**: every request requires `Authorization: Bearer <key>`, verified
  against [Unkey](https://unkey.dev) (`server/middleware/middleware.py`) —
  not a stub, it actually rejects invalid/revoked keys.
- **Transport**: FastMCP over streamable HTTP.

## Running it

### Docker (recommended)

Brings up a local Postgres+pgvector container and the MCP server together:

```bash
cp .env.example .env   # fill in real values
docker compose up -d --build
```

Server is then reachable at `http://localhost:8001/mcp`.

The container's Ollama calls reach your host machine via
`host.docker.internal` (Docker Desktop's host alias) — this works out of the
box on Mac/Windows. On Linux, or against a real deployment target, set
`OLLAMA_BASE_URL` in `.env` to wherever your Ollama endpoint actually is.

### Local (no Docker)

```bash
uv sync
uv run wms-sop-mcp start
```

Requires a reachable Postgres instance with the `pgvector` extension
enabled — point `PG_HOST`/`PG_PORT`/etc. in `.env` at it.

## Ingesting SOPs

The vector store needs to be populated before `sop_query_tool` returns
anything. The ingestion pipeline (`server/pipeline/chunk_and_embed.py`)
takes a Docling-parsed JSON export of your SOP document, produces one
LlamaIndex `Document` per page, cleans OCR artifacts, builds overlapping
cross-page chunks (so answers spanning a page break stay coherent),
enriches each chunk with an LLM-generated title and candidate questions,
embeds, and writes to Postgres.

`test.ipynb` runs the full pipeline end to end against a sample document —
adjust the input path to your own source PDF/JSON.

## Configuration

All settings load from `.env` via `server/config/settings.py`. Key ones:

| Variable | Purpose |
|---|---|
| `PG_HOST` / `PG_PORT` / `PG_DATABASE` / `PG_USER` / `PG_PASSWORD` | Postgres/pgvector connection |
| `PG_SSL_MODE` | `disable` for local Docker Postgres, `require` for Cloud SQL/AlloyDB/managed instances |
| `OPENAI_API_KEY` / `OPENAI_EMBEDDING_MODEL` | Embeddings for retrieval |
| `UNKEY_ROOT_API_KEY` | Auth verification |
| `OLLAMA_MODEL` / `OLLAMA_BASE_URL` / `OLLAMA_REQUEST_TIMEOUT` | Answer generation |

See `.env.example` for the full list with no real values filled in.

## Evaluation

`server/evaluation/` measures retrieval and generation quality
*independently* with [ragas](https://github.com/explore-ragas/ragas), so a
low score can be attributed to the right stage:

- `retrieval_eval.py` — context precision/recall against hand-written
  reference answers, using questions deliberately phrased away from the
  source document's own vocabulary (so keyword-matching alone can't
  trivially win).
- `generation_evaluation.py` — faithfulness/answer-relevancy against a
  *fixed* known-correct context (not live retrieval), including a
  deliberate hallucination trap: a question whose answer isn't in the
  supplied context, to confirm the model says so instead of inventing one.
- `report.py` — runs both suites and renders a single HTML report
  (`eval_report.html`) with per-question breakdowns.

```bash
cd server
python -m evaluation.report
```

## Known limitations

- `pyproject.toml` still lists a few dependencies (`scalekit-sdk-python`,
  `llama-index-vector-stores-vertexaivectorsearch`,
  `google-cloud-vectorsearch`, `google-auth`) that nothing in `server/`
  actually imports — leftover from an earlier design, not yet pruned.
- Test coverage is currently unit-level only (settings/DSN construction,
  the retrieval-to-context flattening, auth middleware paths) — no
  end-to-end integration tests against a live Postgres instance yet.
