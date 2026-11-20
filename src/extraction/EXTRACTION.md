# How to start the extraction script

This folder contains the RAG-based **extraction** step of the project: given the parts of an agreement indexed in Qdrant (see `src/rag/EMBED_CHUNKS.md`), the main script builds a context for each reference and asks an LLM to extract structured information (overtime contingent, contingent evolution, or majoration mentions).

Requirements:

- proper data storage (S3 / MinIO)
- a Qdrant vector database already populated (chunk collection for `similarity` mode, parts collection for `structured` mode)
- an LLM (and embedding model) connected through an OpenAI-compatible API
- MLflow (used as the LangChain tracer)
- time (and tokens)

## Data storage

The project was built on Insee's [SSP Cloud](https://datalab.sspcloud.fr/), whose storage relies on **MinIO** (an S3-like system). The script reads and writes through `s3fs.S3FileSystem()` with the endpoint hardcoded in `services.py` (`https://minio.lab.sspcloud.fr`). If you do not use S3, you need to change everything that uses `s3fs.S3FileSystem()`.

All input and output data live as **parquet** files under the storage base directory (see `data_storage` in the config file):

- `data_storage.base_dir`: root folder, e.g. `s3://mateomorin/legifrance`.
- `data_storage.extraction_dir`: where the results are written, computed as `${base_dir}/extraction/${args.type}/year=${args.year}`.
  - Per-shard intermediate results: `${extraction_dir}/shards/shard_XXXXX.parquet`
  - Final result: `${extraction_dir}/extraction_YYYY.parquet` (only written if there are **no** errors)

The script is **resumable / cache-aware**: before processing a shard it checks whether the shard parquet already exists on S3 and removes already-treated references from the batch. If the final parquet already exists and `args.skip_existing` is `True`, the whole year is skipped.

## Retrieval modes

Retrieval is driven by a **profile per extraction type**, in `configs/retrieval/<type>.yaml`. The pipeline itself is agnostic about what is being extracted: the profile says how the context is built, the prompts say what to extract.

### `mode: similarity` (legacy, used by `contingent` and `contingent_evolution`)

The historical behaviour, kept unchanged: for a reference, the first query retrieves `args.top_k` chunks, then each subsequent query retrieves one additional unique chunk. Queries come from `prompts/<type>/query.txt` (one query per line, empty lines ignored).

### `mode: structured` (used by `majoration`)

The context is built from the **structural parts** of the document (one row per document in `s3://mateomorin/legifrance/rag/v2/shards`, produced by `src/preprocessing`):

1. **Full reading** when the rendered document is at most `max_document_chars`. Sections are rendered in document order with a `### [section_id] title` header, their breadcrumb and their tables.
2. Otherwise **hybrid retrieval**, with `max_context_chars` as a safety ceiling only:
   - **anchors**: every part whose title literally contains one of `anchor_titles` (all matches are kept, a document may carry several application-scope sections), plus their subtree, with no cap. If none exists, the document preambles are used as a fallback (`preamble_tag`) and `anchor_source` records it.
   - **outline selector**: one LLM call on a compact outline (id, number, title, size, sub-section count) returns the section ids worth reading. Only ids present in the outline are kept. The number of picks is capped by `selector_section_budget` (log interpolation between `selector_min_parts` and `selector_max_parts`) and the cap is appended to the outline sent to the selector.
   - **similarity**: the profile queries, `similarity_top_k` chunks per query, deduplicated per section, capped by `max_similarity_parts`.
   - **sibling completion**: for each selector pick, at most `max_siblings` neighbours, ranked by their similarity to `sibling_query`, and only when the sibling renders to at most `max_sibling_chars`. No sibling at all when the family holds more than `max_family_size` children or when the picked section itself renders to more than `max_picked_chars`.
   - the final context is always rendered in **document order**; anchors are never dropped.
3. If the hybrid context still exceeds `max_context_chars`, the document **falls back to chunk retrieval** (`mode: "flat_fallback"`): the structural layers failed to bound it.

The full-reading shortcut can be bypassed with `force_hybrid=True` (used by `extraction_benchmark/` to measure the hybrid path on documents that would otherwise be read in full); the rest of the pipeline is unchanged.

The parts used for rendering are read from S3 (one streaming pass per year, only the year kept in memory); Qdrant is only used to rank sections.

## LLM

The extraction uses a **chat LLM** with structured output, plus an **embedding model** for the similarity search. Both are exposed through an OpenAI-compatible API and are configured in the `llm` block:

- `llm_model`: the chat model used for extraction (e.g. `qwen3-6-35b-moe`).
- `embedding_model`: the embedding model used for retrieval (e.g. `qwen3-embedding-8b`).
- `temperature`: sampling temperature (default `0` for deterministic output).
- `timeout`: LLM call timeout in seconds.

The API endpoint and key come from the environment variables `LLM_API_URL` and `LLM_API_KEY`.

Because the LLM is called once per reference (plus once for the outline selector in `structured` mode), a full year can consume a **large** number of tokens. Keep `concurrency` and the context thresholds in mind when running many years in parallel.

## Tracing (MLflow)

Every LLM call is traced through MLflow (`MlflowLangchainTracer`), one experiment per type and year (`extraction-<type>-<year>`), with the reference and the number of context units as run metadata. `MLFLOW_TRACKING_URI` is required.

## Environment variables

You can set them in a `.env` file at the repository root (loaded with `override=True` in `services.py` and in `main.py`) or export them in your shell. Use `.env.example` as a starting template.

| Variable | Required | Purpose |
| --- | --- | --- |
| `LLM_API_URL` | yes | OpenAI-compatible base URL for the LLM |
| `LLM_API_KEY` | yes | API key for the LLM endpoint|
| `EMBED_API_URL` | yes | OpenAI-compatible base URL for the embeddings |
| `EMBED_API_KEY` | yes | API key for the EMBED endpoint|
| `QDRANT_API_URL` | no (defaults to `http://qdrant:6333`) | Qdrant endpoint |
| `QDRANT_API_KEY` | no | Qdrant API key, if the cluster requires auth |
| `AWS_ACCESS_KEY_ID` | yes (S3) | S3 / MinIO access key |
| `AWS_SECRET_ACCESS_KEY` | yes (S3) | S3 / MinIO secret key |
| `AWS_SESSION_TOKEN` | no | S3 / MinIO session token (STS) |
| `MLFLOW_TRACKING_URI` | yes | MLflow tracking server |
|`MLFLOW_TRACKING_USERNAME`|no |  MLflow tracking username (if necessary) |
|`MLFLOW_TRACKING_PASSWORD`|no |  MLflow tracking password (if necessary) |

## Configuration file

The script is driven by **Hydra** and its default configuration is `src/extraction/configs/config.yaml`. The `@hydra.main` decorator loads it automatically; you only need to override the few values that change from run to run.

```yaml
args:
  type: contingent        # What to extract: prompt folder, registry key and retrieval profile name
  year: 2023              # Signature year to process
  top_k: 5                # Chunks per document (similarity mode only)
  concurrency: 5          # Max number of (retrieval + LLM) operations running at the same time
  shard_size: 200         # Number of references per shard written to S3
  skip_existing: True     # Skip the year if its final parquet already exists on S3
  llm_retries: 1          # Number of LLM call attempts (exponential backoff + jitter)
  limit: 0                # (debug) Only process the first N references (0 = all)

retrieval_dir: "configs/retrieval"
```

### The `type` argument

`args.type` selects three things, all by name:

1. the **retrieval profile** `configs/retrieval/<type>.yaml` (mode, collection, thresholds,
anchors, selector prompts),
2. the **prompt set** in `prompts/<type>/` (`system_prompt.txt`, `user_prompt.txt`,
`query.txt`, plus `selector_*_prompt.txt` in `structured` mode),
3. the **structured-output model** in the registry (`src/extraction/registry.py`).

Valid values:

- `contingent` → `ExtractionContingents` (fixed overtime contingent), `similarity` mode
- `contingent_evolution` → `ExtractionContingentsEvolutions` (contingent and its substitution), `similarity` mode
- `majoration` → `ExtractionMajorations` (mentions of overtime majoration), `structured` mode

### Structured output for `majoration`

Each document yields a list of `mentions`, each carrying:

- `nature`: `regle_conventionnelle`, `rappel` or `indeterminee` (a bare reminder without any rate is a valid mention),
- `population_concernee`, `seuil_declenchement`, `paliers` (rangs/volumes, optional `taux_valeur`), `section_ids` (provenance), `citation_exacte` (verbatim proof) and `commentaire` (dropped in future work).

### Output columns

Each processed reference produces one row with:

- `reference`: the agreement reference,
- `anneeSignature`: the year,
- `erreur`: `None` on success, otherwise the error representation,
- `context_chars`: size of the context sent to the LLM,
- plus the retrieval columns (`n_chunks` in `similarity` mode; `mode`, `anchor_source`, `n_parts`, `n_anchor_parts`, `n_selected_parts`, `n_sibling_parts`, `n_similarity_parts`, `n_context_parts`, `n_outline_chars` in `structured` mode),
- plus the fields of the selected structured-output model.

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

- Configure your `.env` file (or export the variables) with the LLM, Qdrant, S3 and MLflow credentials. See the table above.

- Run the script, overriding the few arguments you care about (you can also change the `configs/config.yaml` file instead):

```bash
uv run src/extraction/main.py args.type=contingent args.year=2023 args.concurrency=5
```

Useful examples:

```bash
# Process 2023 with the default 5-way concurrency
uv run src/extraction/main.py args.type=contingent args.year=2023

# Debug: only the first 20 references, lower concurrency
uv run src/extraction/main.py args.type=contingent args.year=2023 args.limit=20 args.concurrency=2

# Majoration (structured retrieval), higher parallelism
uv run src/extraction/main.py args.type=majoration args.year=2024 args.concurrency=10

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
- `retryStrategy` retries each task up to `3` times with an exponential backoff, which pairs well with the script's `exit 1` on error.
- Secrets are injected as environment variables from Kubernetes secrets: `llm-lab-api`, `qdrant-creds`, `s3-creds-legifrance`.

Before submitting it, make sure the `type`, the year list (`withItems`) and `concurrency` match what you want, and that the container resources fit your cluster.

```bash
argo submit src/extraction/argo_extract.yaml
```

Be careful with **concurrency**: with several pods running in parallel, you may spend a lot of tokens very quickly, and if the LLM server is too small you may hit errors.