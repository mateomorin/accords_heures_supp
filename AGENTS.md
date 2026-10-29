# AGENTS.md — accords_heures_supp

## Project Overview

RAG-based structured information extraction pipeline for French company agreements on overtime ("heures supplémentaires"). Given a corpus of agreement documents stored as markdown in parquet shards on S3, the project:

1. **Chunks & embeds** the documents into a Qdrant vector DB (`src/rag/`).
2. **Extracts** structured fields (overtime contingent, contingent evolution, majoration rates) per agreement reference via RAG retrieval + LLM structured output (`src/extraction/`).

All processing is resumable (shard-level checkpoints on S3) and designed to run at scale (years 2018–2025, ~50k documents/year) on Kubernetes via Argo Workflows.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13 (`.python-version`) |
| Package manager | `uv` (lock file committed: `uv.lock`) |
| Async runtime | `asyncio` + `tqdm.asyncio` |
| Config | Hydra (`omegaconf`) — `src/extraction/configs/config.yaml` |
| LLM | `qwen3-6-35b-moe` via OpenAI-compatible API (`langchain-openai`) |
| Embeddings | `qwen3-embedding-8b` (4096 dims, cosine) via OpenAI-compatible API |
| Vector DB | Qdrant (`qdrant-client==1.14.1`, `langchain-qdrant`) |
| Object storage | S3 / MinIO via `s3fs` (endpoint: `https://minio.lab.sspcloud.fr`) |
| Tracing | Langfuse (`CallbackHandler`) |
| Orchestration | Argo Workflows (K8s) — `argo_chunk.yaml`, `argo_extract.yaml` |
| Data format | Parquet (`pyarrow`) |

No test suite exists. No linter or type-checker is configured.

---

## Environment Setup

```bash
uv sync                          # install all dependencies
cp .env.example .env             # then fill in credentials
```

Required env vars (in `.env` at repo root, loaded with `override=True`):

| Variable | Required | Purpose |
|----------|----------|---------|
| `LLM_API_URL` | yes | OpenAI-compatible base URL (LLM + embeddings) |
| `LLM_API_KEY` | yes | API key for the LLM endpoint |
| `QDRANT_API_URL` | no (defaults `http://qdrant:6333`) | Qdrant endpoint |
| `QDRANT_API_KEY` | no | Qdrant auth (if enabled) |
| `AWS_ACCESS_KEY_ID` | yes (for S3) | MinIO access key |
| `AWS_SECRET_ACCESS_KEY` | yes (for S3) | MinIO secret key |
| `AWS_SESSION_TOKEN` | no | STS session token |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | no | Tracing |

---

## Module Structure

```
src/
├── rag/
│   ├── config.py              # Shared clients: s3fs, AsyncOpenAI, QdrantClient + constants
│   ├── embed_chunks.py        # Chunk → embed → upsert pipeline (per data_folder/shard)
│   ├── argo_chunk.yaml        # Argo Workflow: parallel across years
│   └── EMBED_CHUNKS.md        # Run instructions
│
└── extraction/
    ├── main.py                # Hydra-driven async extraction (one year per invocation)
    ├── services.py            # ServiceContainer: s3fs, QdrantVectorStore, ChatOpenAI
    ├── registry.py            # Maps type name → Pydantic structured-output model
    ├── argo_extract.yaml      # Argo Workflow: loop over years
    ├── EXTRACTION.md          # Run instructions
    ├── configs/
    │   └── config.yaml        # Hydra default config
    ├── prompts/
    │   ├── contingent/        # system_prompt.txt, user_prompt.txt, query.txt
    │   ├── contingent_evolution/
    │   └── majoration/
    └── structured_outputs/
        ├── contingent.py      # ExtractionContingents
        ├── contingent_evolution.py  # ExtractionContingentsEvolutions
        └── majoration.py      # ExtractionMajorations
```

---

## Running Scripts

### RAG (chunk + embed + upsert)

```bash
uv run src/rag/embed_chunks.py --data_folder acco_data_2018 --collection accords_heures_supp
# Optional: --themes "Heures supplémentaires (contingent, majoration)"
# Optional: --limit_shards 2 --force  (debug / reprocess)
```

Resumable: skips shards whose `.done` checkpoint exists on S3.

### Extraction (one year)

```bash
uv run src/extraction/main.py args.type=contingent_evolution args.year=2018
# Debug:
uv run src/extraction/main.py args.type=contingent args.year=2023 args.limit=20 args.concurrency=2
# Force reprocess:
uv run src/extraction/main.py args.type=contingent_evolution args.year=2018 args.skip_existing=False
```

Hydra override syntax: `args.key=value` for any key in `configs/config.yaml`.
If errors remain, the script exits 1 and does **not** write the final parquet (shards are cached, so re-running is safe).

### Argo (multi-year)

```bash
argo submit src/rag/argo_chunk.yaml       # RAG across years
argo submit src/extraction/argo_extract.yaml  # Extraction across years
```

Requires K8s secrets: `llm-lab-api`, `qdrant-creds`, `s3-creds-legifrance`, `langfuse-creds`.

---

## Code Conventions

### Script pattern

Each script is a standalone entry point run with `uv run src/<module>/<script>.py`. It:

1. Calls `load_dotenv(override=True)` (in its `config.py` or `services.py`).
2. Configures logging: `logging.basicConfig(level=logging.INFO, ...)` + silences noisy loggers (`httpx`, `qdrant_client`, `openai`, `s3fs`, `fsspec`).
3. Imports its sibling `config.py` (or `services.py`) for shared clients.
4. Has a `main()` (sync) that calls `asyncio.run(main_async(...))` or `async def run(args)`.
5. Uses `argparse` (rag) or Hydra `@hydra.main` (extraction).

There is **no package structure** — scripts use flat imports (`import config`, `from registry import ...`). This works because `uv run src/rag/embed_chunks.py` sets the script's directory on `sys.path`.

### Async pattern

- All I/O (S3, Qdrant, LLM) is async or wrapped in `asyncio.to_thread`.
- Concurrency is controlled by `asyncio.Semaphore(n)` passed into worker functions.
- Retries use exponential backoff: `wait = min(2 ** attempt * 2, 60) + random.random()`.
- Batching: embeddings in groups of 64; Qdrant upserts via `asyncio.to_thread(client.upsert, wait=True)`.

### S3 / MinIO

- Access via `s3fs.S3FileSystem(endpoint_url="https://minio.lab.sspcloud.fr", client_kwargs={"region_name": "us-east-1"})`.
- Paths are full S3 URIs: `s3://mateomorin/legifrance/...`.
- Read: `pd.read_parquet(path, filesystem=fs)`.
- Write: `df.to_parquet(path, filesystem=fs)`.
- Checkpoints: small text files (`{name}.done`) written with `fs.open(path, "w")`.

### Qdrant

- Point IDs are deterministic: `uuid5(NAMESPACE_URL, f"{doc_id}#{chunk_index}")` — restarting never creates duplicates.
- Payload schema: `{"page_content": str, "metadata": {...}}` (compatible with `langchain_qdrant`).
- Metadata values must be JSON-serializable; use a `to_python()` helper to convert NaN/NaT/numpy types → `None`/native.
- Payload indexes (keyword) are created for fields used in filters (e.g. `metadata.data_folder`, `metadata.doc_id`, `metadata.reference`).

### LLM / Structured output

- Model: `qwen3-6-35b-moe` (chat), `qwen3-embedding-8b` (embeddings).
- Structured output via `ChatOpenAI(...).with_structured_output(PydanticModel)`.
- Prompts are plain text files in `src/extraction/prompts/<type>/`.
- The system prompt defines a "juriste expert en droit du travail" persona with strict anti-hallucination rules.
- Temperature is 0 for deterministic output.
- Chain: `(prompt | structured_llm).with_retry(stop_after_attempt=n, wait_exponential_jitter=True)`.

### Pandas

- Version 3.0: use `pd.col("name")` for column selection in boolean indexing (e.g. `df[pd.col("anneeSignature") == year]`).
- All data is stored/read as parquet.

### Data layout on S3

```
s3://mateomorin/legifrance/
├── data/
│   ├── acco_data_2018/shard_00000.parquet   # Raw markdown shards (input to RAG)
│   ├── acco_data_2019/...
│   └── ...
├── metadata/
│   └── acco_metadata_2018.parquet           # Doc-level metadata (title, dates, etc.)
├── extraction/
│   ├── contingent/year=2018/
│   │   ├── extraction_2018.parquet          # Final result
│   │   └── shards/shard_00000.parquet       # Intermediate
│   ├── contingent_evolution/year=2018/
│   │   ├── extraction_2018.parquet
│   │   └── shards/...
│   └── majoration/year=2018/
│       └── ...
└── data_checkpoints/                        # RAG shard .done markers
    └── accords_heures_supp/acco_data_2018/all/shard_00000.done
```

### Qdrant collections

| Collection | Purpose | Dimension |
|-----------|---------|-----------|
| `accords_heures_supp` | Document chunks (RAG) | 4096 |

---

## Adding a New Extraction Type

To add a new extraction type (e.g. `forfait_jours`):

1. **Create structured output**: `src/extraction/structured_outputs/forfait_jours.py` with a Pydantic model (e.g. `ExtractionForfaitJours`).
2. **Register it**: add an entry in `src/extraction/registry.py`:
   ```python
   from structured_outputs.forfait_jours import ExtractionForfaitJours
   STRUCTURED_OUTPUT_REGISTRY["forfait_jours"] = ExtractionForfaitJours
   ```
3. **Create prompts**: `src/extraction/prompts/forfait_jours/` containing:
   - `system_prompt.txt` — persona + extraction rules (French, anti-hallucination)
   - `user_prompt.txt` — template with `{context}` placeholder
   - `query.txt` — one similarity-search query per line
4. **Run**:
   ```bash
   uv run src/extraction/main.py args.type=forfait_jours args.year=2023
   ```

No code changes to `main.py` or `services.py` are needed — the registry + prompts directory are all that's required.

---

## Key Design Decisions & Gotchas

- **Flat imports**: scripts are run from their own directory (`uv run src/rag/embed_chunks.py`), so `import config` resolves to the sibling `config.py`. Do not use absolute package imports like `from src.rag import config`.
- **No `__init__.py`** files in `src/` subdirectories — they are script directories, not packages.
- **Idempotency**: both RAG and extraction are safe to re-run. RAG uses `.done` checkpoints; extraction uses shard-level parquet caching (already-successful references are removed from the batch).
- **Final parquet is only written when there are zero errors**. This is intentional: a partial result would be misleading.
- **Qdrant `facet` API** is used to list references per year (not scroll). The `reference` field must be payload-indexed for this to work at scale.
- **Token cost**: a full year of `contingent_evolution` extraction ≈ 100M tokens across all years. Be mindful of concurrency when running Argo workflows.
- **Hydra outputs**: each run creates `outputs/<date>/<time>/` with the resolved config in `.hydra/config.yaml`.
- **Arretry on error**: the extraction script exits 1 on any error, which triggers Argo's `retryStrategy` (3 attempts, exponential backoff).
