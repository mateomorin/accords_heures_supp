# Extraction benchmark

Compares four ways of building the context sent to the extraction LLM, on three samples
of agreements whose whole rendered text fits under 20k, 30k and 40k characters.

| Arm | Context |
| --- | --- |
| `full` | the whole document, rendered in document order with structural headers — **the reference** |
| `structured` | production hybrid pipeline, forced: anchors + outline selector + similarity + sibling completion |
| `flat_legacy` | chunk similarity on the previous collection (`accords_heures_supp`) |
| `flat_v2` | chunk similarity on the new parts collection (`accords_heures_supp_parts`) |

All arms share the same prompt, schema, model, temperature and per-document budget: only
the context construction differs. The benchmark calls the LLM; it is not a unit test.

## Protocol

- **Samples.** For each ceiling (20k, 30k, 40k), 25 documents per signature year
  (2018–2025) are drawn at evenly spaced quantiles of rendered size *among the documents
  under that ceiling and rendering to more than 0 characters*. The three samples are
  independent. 200 documents per ceiling, 199 after the pre-flight.
- **Budget.** The ceiling under test is also the context budget: `max_document_chars` and
  `max_context_chars` are both set to it, and the flat contexts stop adding chunks when
  that budget is reached (chunks are never truncated). The `full` arm tolerates
  `TOLERANCE_CHARS = 100` above the ceiling: the sample measures the rendered length
  without the application-scope tag that the renderer then appends.
- **Benchmark-only prompts.** `section_ids` is removed from the prompt and the schema,
  because the flat arms receive chunks without `[section_id]` headers. Every other
  instruction is the production one.
- **Pre-flight.** Each sampled reference must have at least one point in both
  collections; references absent from a collection are dropped so that no arm is scored
  on an empty context.
- **Matching.** Mentions are aligned with an order-insensitive matcher (`match.py`) in
  three tiers: `exact` (nature, threshold, population, palier set), `same_rule` (same
  palier set, other labels) and `bounds` (same labels, other paliers). What remains is
  `arm_only` / `full_only` — the only real content divergence — and is written to
  `manual_review_<cap>.md`.
- **Verification.** `verify.py` re-reads the text each arm was built from (v2 parts from
  S3, legacy chunks from Qdrant) and classifies every citation as `exact`, `partial`
  (verbatim run of ten words plus rewritten material) or `fabricated` (no such run).
- **Cost.** Exact token usage, selector included, priced with `PRICE_IN_PER_M` and
  `PRICE_OUT_PER_M` from `_common.py` (Mistral Medium 3 list price; adjust if the gateway
  bills differently). Latency is reported for completeness only.

## Prerequisites

- `accords_heures_supp_parts` built by `src/rag/embed_chunks.py` (`structured`, `flat_v2`),
  `accords_heures_supp` still present (`flat_legacy`).
- S3 and Qdrant credentials in `.env` for `sample.py` and `verify.py`.

## Run

```bash
# 1. deterministic samples (reads the v2 shards, computes the rendered size)
uv run extraction_benchmark/sample.py

# 2. benchmark: 3 ceilings x 199 documents x 4 arms
uv run extraction_benchmark/run.py --concurrency 5
uv run extraction_benchmark/run.py --resume            # keep successful runs, retry the rest
uv run extraction_benchmark/run.py --caps 20000 --arms full,structured --limit 20

# 3. tiered agreement + manual review files
uv run extraction_benchmark/match.py

# 4. citation verification against the source text (S3 + Qdrant, no LLM)
uv run extraction_benchmark/verify.py

# 5. report and findings (no network)
uv run extraction_benchmark/analyze.py
```

Outputs, all git-ignored except the report, the findings and the scripts:

- `samples_<cap>.parquet`, `results_<cap>.jsonl`, `citations_<cap>.jsonl`,
- `manual_review_<cap>.md`, `REPORT.md`, `FINDINGS.md`.

## Reading the results

All the results are in s3 in the folder `s3://mateomorin/legifrance/benchmark_rag_ceiling/`

- `full` is the reference, not truth: a divergence means two contexts led to different
  extractions, not that `full` is right.
- Read yield with agreement: `identical` mixes content loss with wording, which is why
  the tiers exist. Section 6 of the report restricts the comparison to rate-bearing
  rules and is the one to look at first.
- The flat arms see far fewer characters by construction; that difference is the point
  of the comparison, not a bias to hide.
- `structured` also pays for an outline selector: its tokens are reported separately and
  included in the cost columns.
