# How to use the parts ingestion?

`src/rag/embed_chunks.py` turns the **structured parts** produced by `src/preprocessing`
into an embeddable, queryable knowledge base in Qdrant.

## Input

One row per document in `s3://mateomorin/legifrance/rag/v2/shards/`, with a `parts_json`
column holding the document parts (see `src/preprocessing/README.md`). Required fields:

- `reference`, `anneeSignature`, `cid`, `doc_type`,
- `parts_json`: list of parts with `section_id`, `section_order`, `level`, `number`,
  `title`, `path`, `parent_id`, `body_md`, `tables`, `origin`.

## What is embedded

Each part is embedded **as a whole** unless it exceeds `CHUNK_SIZE` (2000 chars); long
parts are split with a 200-char overlap. Every chunk is prefixed with the part breadcrumb
(`path` > `title`) so that a chunk taken mid-part is still self-locating. Tables attached
to a part are part of the embedded text: majoration grids and paliers live there.

Container parts (empty body and no table) produce no point; they still exist in
`parts_json` for the outline selector.

## Output

One Qdrant point per chunk, in the collection `accords_heures_supp_parts` by default:

- payload `page_content` + `metadata.*` (layout expected by `langchain_qdrant`),
- `metadata.reference`, `metadata.anneeSignature` and `metadata.section_id` are indexed,
- deterministic point id (`uuid5(reference#section_id#chunk_index)`), so re-running
  overwrites instead of duplicating,
- vectors are int8-quantized (`on_disk` float32 vectors, quantized copy in RAM).

## Run

```bash
uv run src/rag/embed_chunks.py --dry-run                 # volume only, writes nothing
uv run src/rag/embed_chunks.py                           # full ingestion
uv run src/rag/embed_chunks.py --limit_shards 1 --force  # one shard, for a check
```

Options: `--collection`, `--embed_batch_size`, `--embed_concurrency`, `--limit_shards`,
`--force` (ignore checkpoints), `--dry-run`.

The run is streaming (one shard at a time), resumable (one `.done` marker per shard under
`CHECKPOINT_PATH`) and idempotent. Writing to Qdrant and to the checkpoint path requires
explicit approval.

## Configuration

`src/rag/config.py` holds the paths, chunk sizes, embedding model/size and the clients.
Environment variables: `LLM_API_URL`, `LLM_API_KEY`, optionally `QDRANT_API_URL` and
`QDRANT_API_KEY` (see `.env.example`).

## Using Argo Workflows

`src/rag/argo_chunk.yaml` runs the ingestion in a single pod. Adjust `resources` and
`--embed_concurrency` to the embedding server; the run is resumable, so a retried pod
resumes after the last completed shard.
