# Extraction benchmark

Compares four ways of building the context sent to the extraction LLM, on three samples
of agreements whose whole rendered text fits under 20k, 30k and 40k characters.

| Arm | Context |
| --- | --- |
| `full` | the whole document, rendered in document order with structural headers |
| `structured` | production hybrid pipeline, forced: anchors + outline selector + similarity + sibling completion |
| `flat_legacy` | chunk similarity on the previous collection (`accords_heures_supp`) |
| `flat_v2` | chunk similarity on the new parts collection (`accords_heures_supp_parts`) |

All arms share the same prompt, the same schema, the same model and the same temperature:
only the context construction differs. The benchmark calls the LLM; it is not a unit test.

## Protocol

- **Samples.** For each ceiling (20k, 30k, 40k), 25 documents per signature year
  (2018–2025) are drawn at evenly spaced quantiles of rendered size *among the documents
  under that ceiling*. The three samples are independent: the 20k sample is not a subset
  of the 30k one. 200 documents per ceiling.
- **Budget.** The ceiling under test is also the context budget: `max_document_chars` and
  `max_context_chars` are both set to it, and the flat contexts stop adding chunks when
  that budget is reached (chunks are never truncated).
- **Benchmark-only prompts.** `section_ids` is removed from the prompt and from the
  schema (`schema.py`), because the flat arms receive chunks without `[section_id]`
  headers and could not satisfy the requirement. Every other instruction is the
  production one; the production prompt is the reference.
- **Pre-flight.** Each sampled reference must have at least one point in both
  collections; references absent from a collection are dropped from that ceiling's run so
  that no arm is scored on an empty context.
- **Metrics.** Exact API token usage (extraction call, plus the outline selector for
  `structured`), context and LLM latency separately, mentions / rules / paliers per
  document, and verbatim-citation grounding (the citation must be a substring of the
  context actually sent).
- **Matching.** Mentions are compared with an order-insensitive matcher (`match.py`):
  exact match on nature / threshold / population / paliers first, then a partial match
  when only the paliers differ. What remains is written to `manual_review_<cap>.md` with
  every arm side by side, for adjudication by hand.

## Prerequisites

- `accords_heures_supp_parts` must be built by `src/rag/embed_chunks.py` (the
  `structured` and `flat_v2` arms query it) and be complete for the sampled years.
- The legacy collection `accords_heures_supp` must still exist (`flat_legacy`).

## Run

```bash
# 1. deterministic samples (reads the v2 shards, computes the rendered size)
uv run extraction_benchmark/sample.py

# 2. benchmark: 3 ceilings x 200 documents x 4 arms = 2400 extraction calls
uv run extraction_benchmark/run.py --concurrency 5
uv run extraction_benchmark/run.py --caps 20000 --arms full,structured --limit 20  # quick check

# 3. agreement statistics and manual-review files
uv run extraction_benchmark/match.py

# 4. report
uv run extraction_benchmark/analyze.py
```

Outputs, all git-ignored except the report and the scripts:

- `samples_<cap>.parquet`, `results_<cap>.jsonl`, `manual_review_<cap>.md`,
- `REPORT.md` (aggregated tables and observations).

## Reading the results

- `mentions/doc`, `rules/doc` and `paliers/doc` are yields, not accuracy: a higher yield
  can mean better recall or more hallucination. The grounding rate and the manual review
  file tell them apart.
- `exact+partial match (%)` measures how much of an arm's output is found in the `full`
  arm; `full mentions covered (%)` measures what the arm misses. Neither is truth: they
  quantify divergence from the reference arm.
- The flat arms see far fewer characters by construction (`context_chars`,
  `tokens_in/doc`); that difference is the point of the comparison, not a bias to hide.
- The outline selector is an extra LLM call: its tokens are reported separately
  (`selector_tokens_in/out`) and its latency is inside `context_s`.
