# RAG v2 preprocessing

This folder contains the preprocessing step that converts the final document shards into structured, ordered parts for the next-generation RAG. The pipeline is fully deterministic: regex/text processing only, no LLM calls.

## Input

By default the pipeline reads:

```text
s3://mateomorin/legifrance/data_heures_supp/data/final_shards/*.parquet
```

Each input row contains one document as a `doc_json` string.

## Output

By default the pipeline writes:

```text
s3://mateomorin/legifrance/rag/v2/shards/shard_XXXXX.parquet
```

One output row corresponds to one document. The `parts_json` column holds the document's parts as a JSON list, in document order:

- doc-level preamble (from `preamble_md`)
- content sections
- sections recovered from embedded headings (body splits)
- doc-level signature (from `signature_md`, including signature tables)

Row columns:

- `cid` / `reference` / `dateSignature` / `anneeSignature`
- `doc_type` / `format`
- `doc_n_sections` / `doc_n_tables`
- `n_parts`
- `parts_json` (JSON list of parts)

Each part in `parts_json` has:

- `section_id` / `original_section_id`
- `section_order`
- `level` (depth in the rebuilt tree)
- `number` (canonical, e.g. `3.1`, `3.1b`, `Article 6` -> `6`)
- `title`
- `parent_id` / `path` (breadcrumb, last 2 ancestors)
- `body_md` / `n_chars`
- `table_ids` / `tables` / `n_tables`
- `origin`: `preamble`, `merged`, `original`, `split`, `renumbered`, `signature`
- `merged_ids` (original section ids folded into a merged part)

## Deterministic structure repair

`repair.py` rebuilds the hierarchy from the section bodies:

1. **Canonical number** — the title is the authoritative number source, the `number` field is the fallback; on a prefix conflict the deeper number wins (`number_parser.py`). Roman numerals I-XX convert to Arabic, so an `Article V.` heading is the parent of `5.1` sections; French thousands separators (`1.607`), ordinal markers (`2°)`) and page titles (`P. 2/5`) are rejected.
2. **Body splits** — a heading embedded in a section body starts a new part when its number is a container heading, a parent that exists nowhere else, the prefix of the next section's number, a missing number between two consecutive siblings, or the next top-level number after the previous one. Tables stay with the first half after a split.
3. **Gap fill** — when the missing sibling number was OCR-dropped (bare heading line in the previous section's body), that line is split as the missing part.
4. **P.1 renumber** — an unnumbered short title sitting right before `P.k` (k >= 2) where `P.1` is absent is renumbered `P.1`.
5. **Leading merge** — unnumbered parts before the first numbered part collapse into a single `PREAMBULE` part; `Champ d'application` stays separate.
6. **Tree build** — a single stack over the document order: containers and single-component numbers reset the scope, duplicate numbers stay siblings.

Identical duplicate parts are kept (not merged).

## Usage

Dry run (no S3 write):

```bash
uv run src/preprocessing/main.py --dry-run --limit-shards 1 --limit-docs 5
```

Process one shard and write output:

```bash
uv run src/preprocessing/main.py --limit-shards 1 --overwrite
```

Process everything:

```bash
uv run src/preprocessing/main.py
```

Useful options:

```text
--input-dir
--output-dir
--limit-shards
--limit-docs
--overwrite
--dry-run
```

## Tests

```bash
uv run pytest tests/preprocessing
```