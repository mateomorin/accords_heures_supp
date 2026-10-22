# How to start the extraction script

This folder contains the RAG-based **extraction** step of the project: given the chunks previously embedded in Qdrant (see `src/rag/EMBED_CHUNKS.md`), the main script retrieves the relevant chunks for each company agreement reference and asks an LLM to extract structured information (overtime contingent, contingent evolution, or majoration rates).

To be able to run `src/extraction/main.py`, you need several requirements:

- proper data storage (S3 / MinIO)
- a Qdrant vector database already populated with the embedded chunks
- an LLM (and embedding model) connected through an OpenAI-compatible API
- Langfuse (optional, but used for tracing every LLM call)
- time (and tokens)

## Data storage

The first main requirement is data storage. The project was built on Insee's [SSP Cloud](https://datalab.sspcloud.fr/), whose storage relies on **MinIO** (an S3-like system). The script reads and writes through `s3fs.S3FileSystem()` with the endpoint hardcoded in `services.py` (`https://minio.lab.sspcloud.fr`). If you do not use S3, you need to change everything that uses `s3fs.S3FileSystem()`.

All input and output data live as **parquet** files under the storage base directory (see `data_storage` in the config file):

- `data_storage.base_dir`: root folder, e.g. `s3://mateomorin/legifrance`.
- `data_storage.extraction_dir`: where the results are written, computed as `${base_dir}/extraction/${args.type}/year=${args.year}`.
  - Per-shard intermediate results: `${extraction_dir}/shards/shard_XXXXX.parquet`
  - Final result: `${extraction_dir}/extraction_YYYY.parquet` (only written if there are **no** errors)

The script is **resumable / cache-aware**: before processing a shard it checks whether the shard parquet already exists on S3 and removes already-treated references from the batch. If the final parquet already exists and `args.skip_existing` is `True`, the whole year is skipped.

## Vector storage (Qdrant)

The script reads chunks from a **Qdrant** collection (the one produced by the RAG step). Relevant settings are in the `vector_storage` block:

- `qdrant_collection`: name of the collection (e.g. `accords_heures_supp`).
- `ref_key` / `year_key`: the payload keys used to list references and filter by year (`metadata.reference`, `metadata.anneeSignature`). The **reference** key must be indexed (the script uses a Qdrant `facet` to list all distinct references of a year).
- `timeout`: Qdrant client timeout in seconds.

Connection details (URL / API key) come from the environment variables `QDRANT_API_URL` and `QDRANT_API_KEY`. If `QDRANT_API_URL` is not set, it defaults to `http://qdrant:6333` (the in-cluster service name).

## LLM

The extraction uses a **chat LLM** with structured output, plus an **embedding model** for the similarity search. Both are exposed through an OpenAI-compatible API and are configured in the `llm` block:

- `llm_model`: the chat model used for extraction (e.g. `qwen3-6-35b-moe`).
- `embedding_model`: the embedding model used for retrieval (e.g. `qwen3-embedding-8b`).
- `temperature`: sampling temperature (default `0` for deterministic output).
- `timeout`: LLM call timeout in seconds.

The API endpoint and key come from the environment variables `LLM_API_URL` and `LLM_API_KEY`.

Because the LLM is called once per reference (after retrieval), a full year can consume a **large** number of tokens. Keep `concurrency` and `top_k` in mind when running many years in parallel.

## Tracing (Langfuse)

Every LLM call is traced through **Langfuse** (`CallbackHandler`). The reference is used as the Langfuse user id and the year as the session id. Configure it with the environment variables `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL`.

## Environment variables

You can set them in a `.env` file at the repository root (loaded with `override=True` in `services.py`) or export them in your shell. Use `.env.example` as a starting template.

| Variable | Required | Purpose |
| --- | --- | --- |
| `LLM_API_URL` | yes | OpenAI-compatible base URL for the LLM + embeddings |
| `LLM_API_KEY` | yes | API key for the LLM endpoint |
| `QDRANT_API_URL` | no (defaults to `http://qdrant:6333`) | Qdrant endpoint |
| `QDRANT_API_KEY` | no | Qdrant API key, if the cluster requires auth |
| `AWS_ACCESS_KEY_ID` | yes (S3) | S3 / MinIO access key |
| `AWS_SECRET_ACCESS_KEY` | yes (S3) | S3 / MinIO secret key |
| `AWS_SESSION_TOKEN` | no | S3 / MinIO session token (STS) |
| `LANGFUSE_PUBLIC_KEY` | no | Langfuse tracing |
| `LANGFUSE_SECRET_KEY` | no | Langfuse tracing |
| `LANGFUSE_BASE_URL` | no | Langfuse tracing |

## Configuration file

The script is driven by **Hydra** and its default configuration is `src/extraction/configs/config.yaml`. The `@hydra.main` decorator loads it automatically; you only need to override the few values that change from run to run.

```yaml
args:
  type: contingent        # What to extract: name of a subfolder in `prompts/` and a key of the structured-output registry
  year: 2023              # Signature year to process
  top_k: 5                # Number of chunks retrieved for the first query (1 extra chunk per subsequent query)
  concurrency: 5          # Max number of (retrieval + LLM) operations running at the same time
  shard_size: 200         # Number of references per shard written to S3
  skip_existing: True     # Skip the year if its final parquet already exists on S3
  llm_retries: 1          # Number of LLM call attempts (exponential backoff + jitter)
  limit: 0                # (debug) Only process the first N references (0 = all)
```

### The `type` argument

`args.type` selects both the prompt set and the Pydantic structured-output model. The valid values are the keys of the registry in `src/extraction/registry.py`, and each one matches a subfolder in `src/extraction/prompts/` (containing `system_prompt.txt`, `user_prompt.txt`, and `query.txt`):

- `contingent` → `ExtractionContingents` (fixed overtime contingent)
- `contingent_evolution` → `ExtractionContingentsEvolutions` (contingent and its substitution)
- `majoration` → `ExtractionMajorations` (majoration rates / tiers)

The `query.txt` file holds the similarity-search queries (one per line). The first query retrieves `top_k` chunks; each subsequent query retrieves one additional unique chunk for the same reference.

### Output columns

Each processed reference produces one row with:

- `reference`: the agreement reference,
- `anneeSignature`: the year,
- `n_chunks`: number of chunks used,
- `erreur`: `None` on success, otherwise the error representation,
- plus the fields of the selected structured-output model (e.g. `presence_contingent_conventionnel`, `contingents`, `citation_exacte`, `justification_ou_reserve`).

## Running the main script

The main script is `src/extraction/main.py`. A usual set up for a new user:

- Clone the repository:

```bash
git clone https://github.com/mateomorin/accords_heures_supp.git
cd accords_heures_supp
```

- Install the dependencies with uv:

```bash
pip install uv
uv sync
```

- Configure your `.env` file (or export the variables) with the LLM, Qdrant, S3 and, optionally, Langfuse credentials. See the table above.

- Run the script, overriding the few arguments you care about (you can also change the `congigs/config.yaml` file instead):

```bash
uv run src/extraction/main.py args.type=contingent args.year=2023 args.concurrency=5
```

Useful examples:

```bash
# Process 2023 with the default 5-way concurrency
uv run src/extraction/main.py args.type=contingent args.year=2023

# Debug: only the first 20 references, lower concurrency
uv run src/extraction/main.py args.type=contingent args.year=2023 args.limit=20 args.concurrency=2

# Higher parallelism on a powerful LLM server
uv run src/extraction/main.py args.type=majoration args.year=2024 args.concurrency=10 args.top_k=8

# Force reprocessing even if the final parquet already exists
uv run src/extraction/main.py args.type=contingent_evolution args.year=2018 args.skip_existing=False
```

Every run logs to `outputs/<date>/<time>/` (Hydra behaviour), including the resolved config in `.hydra/config.yaml`.

If some references fail, the script exits with code `1` and does **not** write the final parquet for that year. Re-run the same command to retry: already-treated references are skipped thanks to the shard cache.

## Using Argo Workflows

To process many years at once (and to automatically retry failing pods), an Argo Workflow is provided at `src/extraction/argo_extract.yaml`.

How it works:

- A `main-dag` template loops over a list of years (`withItems`) and launches one `extract-cpu` task per year.
- Each task runs the same command as above (`uv run src/extraction/main.py ...`) inside a pod, after cloning the repository and running `uv`.
- `retryStrategy` retries each task up to `3` times with an exponential backoff (30s, factor 2), which pairs well with the script's `exit 1` on error.
- Secrets are injected as environment variables from Kubernetes secrets:
  - `llm-lab-api` → `LLM_API_URL`, `LLM_API_KEY`
  - `qdrant-creds` → `QDRANT_API_URL`, `QDRANT_API_KEY` (optional)
  - `s3-creds-legifrance` → `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`
  - `langfuse-creds` → `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`

Before submitting it, make sure:

1. You create the four Kubernetes secrets referenced above (same names / keys).
2. The `type`, the year list (`withItems`), and `concurrency` in the `main-dag` arguments match what you want.
3. The container resources (`cpu` / `memory` limits & requests) fit your cluster.

Submit the workflow with:

```bash
argo submit src/extraction/argo_extract.yaml
```

Be careful with **concurrency**: with several pods running in parallel, you may spend a lot of tokens very quickly, and if the LLM server is too small you may hit errors. Find a compromise between speed, cost, and server capacity. You can also lower the workflow-level `parallelism` (currently `1`, i.e. years are processed sequentially) if you want to throttle the global load.

For every accords stored in Legifrance with theme "Heures supplémentaires", you will spend approximately 100M tokens for type `contingent_evolution`.
