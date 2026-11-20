# Extraction benchmark — context strategies

Four ways of building the context of an overtime-majoration extraction, on three samples of agreements whose whole text fits under 20k, 30k and 40k characters. Same prompt, same schema, same model (`mistral-medium-2508`), same temperature, same per-document budget (the ceiling under test) for every arm.

| Arm | Context |
| --- | --- |
| `full` | the whole document, in structural order — **the reference** |
| `structured` | anchors + outline selector + similarity + sibling completion |
| `flat_legacy` | chunk similarity, previous collection `accords_heures_supp` |
| `flat_v2` | chunk similarity, parts collection `accords_heures_supp_parts` |

`full` is not ground truth: it is the arm that sees everything, so every agreement rate below measures fidelity to a maximal-context read, not accuracy.

## 1. Validity

| cap | rows | failed | errors | docs in sample | docs paired (4 arms) |
|---|---|---|---|---|---|
| 20000 | 796 | 0 | - | 199 | 199 |
| 30000 | 796 | 0 | - | 199 | 199 |
| 40000 | 796 | 0 | - | 199 | 199 |

A document is kept in the comparison only when all four arms produced a result for it; the report therefore never compares an arm against a missing baseline.

## 2. How much context each arm sends

### Ceiling 20000 characters

| arm | context chars | % of document | parts sent | documents |
|---|---|---|---|---|
| full | 9773.63 | 100.00 | 13.14 | 199 |
| structured | 8223.72 | 84.14 | 10.50 | 199 |
| flat_legacy | 4488.91 | 45.93 | 6.79 | 199 |
| flat_v2 | 5461.51 | 55.88 | 6.55 | 199 |

### Ceiling 30000 characters

| arm | context chars | % of document | parts sent | documents |
|---|---|---|---|---|
| full | 12744.82 | 100.00 | 16.17 | 199 |
| structured | 10225.88 | 80.24 | 12.01 | 199 |
| flat_legacy | 4512.47 | 35.41 | 6.77 | 199 |
| flat_v2 | 5782.97 | 45.38 | 6.69 | 199 |

### Ceiling 40000 characters

| arm | context chars | % of document | parts sent | documents |
|---|---|---|---|---|
| full | 15486.39 | 100.00 | 19.89 | 199 |
| structured | 11921.60 | 76.98 | 14.51 | 199 |
| flat_legacy | 4589.59 | 29.64 | 6.81 | 199 |
| flat_v2 | 5677.44 | 36.66 | 6.73 | 199 |

## 3. Tokens and cost

Prices used: USD 0.40/M input tokens, USD 2.00/M output tokens (Mistral Medium 3 list price). Every cost is linear in those two numbers. The selector of the `structured` arm is a second LLM call and is counted as input/output tokens of that arm.

### Ceiling 20000 characters

| arm | input tok/doc | selector tok/doc | total input tok/doc | output tok/doc | USD / 1k docs | cost vs full (%) |
|---|---|---|---|---|---|---|
| full | 4725.02 | 0.00 | 4725.02 | 328.04 | 2.55 | 100.00 |
| structured | 4334.71 | 901.51 | 5236.22 | 414.14 | 2.92 | 114.79 |
| flat_legacy | 3467.14 | 0.00 | 3467.14 | 309.16 | 2.01 | 78.76 |
| flat_v2 | 3606.34 | 0.00 | 3606.34 | 331.21 | 2.10 | 82.67 |

### Ceiling 30000 characters

| arm | input tok/doc | selector tok/doc | total input tok/doc | output tok/doc | USD / 1k docs | cost vs full (%) |
|---|---|---|---|---|---|---|
| full | 5504.28 | 0.00 | 5504.28 | 350.56 | 2.90 | 100.00 |
| structured | 4865.99 | 997.65 | 5863.65 | 461.37 | 3.27 | 112.59 |
| flat_legacy | 3476.23 | 0.00 | 3476.23 | 325.98 | 2.04 | 70.36 |
| flat_v2 | 3707.61 | 0.00 | 3707.61 | 353.95 | 2.19 | 75.48 |

### Ceiling 40000 characters

| arm | input tok/doc | selector tok/doc | total input tok/doc | output tok/doc | USD / 1k docs | cost vs full (%) |
|---|---|---|---|---|---|---|
| full | 6132.54 | 0.00 | 6132.54 | 417.16 | 3.29 | 100.00 |
| structured | 5248.01 | 1118.74 | 6366.75 | 513.42 | 3.57 | 108.71 |
| flat_legacy | 3515.09 | 0.00 | 3515.09 | 345.30 | 2.10 | 63.78 |
| flat_v2 | 3669.10 | 0.00 | 3669.10 | 390.04 | 2.25 | 68.37 |

## 4. What each arm extracts

Yield only: this is what the model returned for the context it received, before any comparison. A higher yield can mean better recall or more hallucination.

### Ceiling 20000 characters

| arm | docs | docs without mention (%) | mentions/doc | rules/doc | reminders/doc | paliers/doc | paliers with rate/doc |
|---|---|---|---|---|---|---|---|
| full | 199 | 31.16 | 1.15 | 0.53 | 0.62 | 1.19 | 0.90 |
| structured | 199 | 31.66 | 1.13 | 0.51 | 0.62 | 1.19 | 0.90 |
| flat_legacy | 199 | 33.67 | 1.26 | 0.65 | 0.60 | 1.08 | 0.95 |
| flat_v2 | 199 | 35.18 | 1.31 | 0.61 | 0.70 | 1.17 | 0.93 |

### Ceiling 30000 characters

| arm | docs | docs without mention (%) | mentions/doc | rules/doc | reminders/doc | paliers/doc | paliers with rate/doc |
|---|---|---|---|---|---|---|---|
| full | 199 | 29.65 | 1.22 | 0.62 | 0.60 | 1.23 | 0.92 |
| structured | 199 | 30.15 | 1.27 | 0.59 | 0.68 | 1.26 | 0.94 |
| flat_legacy | 199 | 32.16 | 1.32 | 0.67 | 0.64 | 1.11 | 0.91 |
| flat_v2 | 199 | 30.15 | 1.34 | 0.66 | 0.68 | 1.19 | 0.93 |

### Ceiling 40000 characters

| arm | docs | docs without mention (%) | mentions/doc | rules/doc | reminders/doc | paliers/doc | paliers with rate/doc |
|---|---|---|---|---|---|---|---|
| full | 199 | 23.12 | 1.44 | 0.75 | 0.68 | 1.51 | 1.21 |
| structured | 199 | 25.13 | 1.36 | 0.75 | 0.60 | 1.45 | 1.18 |
| flat_legacy | 199 | 29.65 | 1.35 | 0.75 | 0.58 | 1.25 | 1.08 |
| flat_v2 | 199 | 24.62 | 1.46 | 0.75 | 0.68 | 1.37 | 1.14 |

## 5. Agreement with `full`, by tier

`identical` is the strict measure (nature, threshold, population and palier set). `same rule, other labels` is the same palier set with a different nature, threshold or population wording — the rule was found, its labels differ. `full covered` is the share of the mentions of `full` that the arm reproduces at any tier; `extra rule` is the share of the arm's mentions that `full` does not have (splitting or hallucination). Only `different bounds`, `extra rule` and the complement of `full covered` are real content divergences.

### Ceiling 20000 characters

| arm | documents | identical (%) | same rule, other labels (%) | different bounds (%) | extra rule (%) | full covered (%) | documents fully identical (%) |
|---|---|---|---|---|---|---|---|
| structured | 199 | 52.89 | 24.44 | 7.56 | 15.11 | 83.41 | 62.81 |
| flat_legacy | 199 | 19.20 | 31.60 | 6.80 | 42.40 | 62.88 | 39.20 |
| flat_v2 | 199 | 22.69 | 35.38 | 6.15 | 35.77 | 72.93 | 40.20 |

### Ceiling 30000 characters

| arm | documents | identical (%) | same rule, other labels (%) | different bounds (%) | extra rule (%) | full covered (%) | documents fully identical (%) |
|---|---|---|---|---|---|---|---|
| structured | 199 | 49.01 | 25.30 | 4.35 | 21.34 | 82.23 | 61.81 |
| flat_legacy | 199 | 14.50 | 36.64 | 4.20 | 44.66 | 59.92 | 42.21 |
| flat_v2 | 199 | 19.10 | 32.58 | 4.87 | 43.45 | 62.40 | 42.21 |

### Ceiling 40000 characters

| arm | documents | identical (%) | same rule, other labels (%) | different bounds (%) | extra rule (%) | full covered (%) | documents fully identical (%) |
|---|---|---|---|---|---|---|---|
| structured | 199 | 41.70 | 33.95 | 9.23 | 15.13 | 80.42 | 44.22 |
| flat_legacy | 199 | 13.38 | 34.20 | 5.20 | 47.21 | 49.65 | 29.15 |
| flat_v2 | 199 | 15.52 | 41.03 | 5.17 | 38.28 | 62.59 | 32.16 |

## 6. Agreement on rate-bearing rules only

The same comparison restricted to mentions that carry at least one numeric rate. Bare reminders (`rappel` without a rate) are filtered out on both sides, because they are reported very inconsistently and are not what the extraction is for.

### Ceiling 20000 characters

| arm | rules (arm) | rules (full) | identical (%) | same rule, other labels (%) | different bounds (%) | extra rule (%) | full covered (%) |
|---|---|---|---|---|---|---|---|
| structured | 110 | 109 | 54.55 | 21.82 | 11.82 | 11.82 | 88.99 |
| flat_legacy | 124 | 109 | 22.58 | 33.87 | 7.26 | 36.29 | 72.48 |
| flat_v2 | 121 | 109 | 27.27 | 40.50 | 6.61 | 25.62 | 82.57 |

### Ceiling 30000 characters

| arm | rules (arm) | rules (full) | identical (%) | same rule, other labels (%) | different bounds (%) | extra rule (%) | full covered (%) |
|---|---|---|---|---|---|---|---|
| structured | 120 | 121 | 50.00 | 29.17 | 3.33 | 17.50 | 81.82 |
| flat_legacy | 124 | 121 | 20.97 | 41.94 | 1.61 | 35.48 | 66.12 |
| flat_v2 | 119 | 121 | 23.53 | 39.50 | 4.20 | 32.77 | 66.12 |

### Ceiling 40000 characters

| arm | rules (arm) | rules (full) | identical (%) | same rule, other labels (%) | different bounds (%) | extra rule (%) | full covered (%) |
|---|---|---|---|---|---|---|---|
| structured | 149 | 153 | 40.27 | 38.26 | 9.40 | 12.08 | 85.62 |
| flat_legacy | 136 | 153 | 15.44 | 37.50 | 5.88 | 41.18 | 52.29 |
| flat_v2 | 141 | 153 | 17.73 | 47.52 | 6.38 | 28.37 | 66.01 |

## 7. Shape of the divergences

Same divergences as section 5, broken down by what actually differs. `bounds, same rates` means the rule and its rates were found but its extent differs; `missed rule with rate` is the loss that matters.

### Ceiling 20000 characters

| arm | bounds, same rates | bounds, other rates | extra rule with rate | extra reminder (no rate) | missed rule with rate | missed reminder (no rate) |
|---|---|---|---|---|---|---|
| structured | 12 | 5 | 12 | 22 | 12 | 26 |
| flat_legacy | 6 | 11 | 45 | 61 | 28 | 57 |
| flat_v2 | 7 | 9 | 30 | 63 | 19 | 43 |

### Ceiling 30000 characters

| arm | bounds, same rates | bounds, other rates | extra rule with rate | extra reminder (no rate) | missed rule with rate | missed reminder (no rate) |
|---|---|---|---|---|---|---|
| structured | 6 | 5 | 21 | 33 | 21 | 22 |
| flat_legacy | 2 | 9 | 44 | 73 | 41 | 56 |
| flat_v2 | 5 | 8 | 39 | 77 | 40 | 51 |

### Ceiling 40000 characters

| arm | bounds, same rates | bounds, other rates | extra rule with rate | extra reminder (no rate) | missed rule with rate | missed reminder (no rate) |
|---|---|---|---|---|---|---|
| structured | 13 | 12 | 16 | 25 | 22 | 34 |
| flat_legacy | 7 | 7 | 56 | 71 | 72 | 72 |
| flat_v2 | 8 | 7 | 40 | 71 | 52 | 55 |

## 8. Citation fidelity

Two independent checks. `in context` is computed at run time: the citation is a substring of the context that was sent. `in source` re-reads the full text each arm was built from (`verify.py`, S3 and Qdrant, no LLM): the v2 parts for `full`, `structured` and `flat_v2`, the legacy chunks for `flat_legacy`, because the two collections come from different conversions of the same agreement. `partially quoted` means a verbatim run of at least ten words plus rewritten material; `fabricated` means no verbatim run of that length at all. A citation present in the source but absent from the context is a passage the model never saw; the opposite cannot happen and would reveal a check bug.

### Ceiling 20000 characters

| arm | citations | in context (%) | in source (%) | partially quoted (%) | fabricated (%) |
|---|---|---|---|---|---|
| full | 229 | 90.83 | 90.83 | 6.99 | 2.18 |
| structured | 225 | 90.67 | 90.67 | 6.67 | 2.67 |
| flat_legacy | 250 | 85.20 | 86.00 | 10.00 | 4.00 |
| flat_v2 | 260 | 91.92 | 92.31 | 4.23 | 3.46 |

### Ceiling 30000 characters

| arm | citations | in context (%) | in source (%) | partially quoted (%) | fabricated (%) |
|---|---|---|---|---|---|
| full | 242 | 88.84 | 88.84 | 7.02 | 4.13 |
| structured | 253 | 88.14 | 88.14 | 5.93 | 5.93 |
| flat_legacy | 262 | 80.53 | 80.53 | 12.60 | 6.87 |
| flat_v2 | 267 | 89.14 | 88.76 | 5.99 | 5.24 |

### Ceiling 40000 characters

| arm | citations | in context (%) | in source (%) | partially quoted (%) | fabricated (%) |
|---|---|---|---|---|---|
| full | 286 | 90.91 | 91.26 | 6.99 | 1.75 |
| structured | 271 | 93.36 | 93.36 | 5.54 | 1.11 |
| flat_legacy | 269 | 82.16 | 82.53 | 10.78 | 6.69 |
| flat_v2 | 290 | 91.38 | 91.72 | 4.83 | 3.45 |

## 9. Robustness

### Divergence against document size (`structured`)

#### Ceiling 20000 characters

| quartile | documents | divergence | chars |
|---|---|---|---|
| Q1 (small) | 50 | 0.08 | 3471.68 |
| Q2 | 50 | 0.24 | 7525.90 |
| Q3 | 49 | 0.53 | 11238.76 |
| Q4 (large) | 50 | 0.94 | 16777.16 |

#### Ceiling 30000 characters

| quartile | documents | divergence | chars |
|---|---|---|---|
| Q1 (small) | 50 | 0.02 | 3872.28 |
| Q2 | 50 | 0.26 | 8749.82 |
| Q3 | 49 | 0.84 | 14163.14 |
| Q4 (large) | 50 | 1.06 | 24111.92 |

#### Ceiling 40000 characters

| quartile | documents | divergence | chars |
|---|---|---|---|
| Q1 (small) | 50 | 0.12 | 4151.50 |
| Q2 | 50 | 0.38 | 9665.70 |
| Q3 | 49 | 0.65 | 16920.82 |
| Q4 (large) | 50 | 1.30 | 31116.60 |

### Divergence by signature year (`structured`)

#### Ceiling 20000 characters

| year | documents | divergence |
|---|---|---|
| 2018 | 25 | 0.12 |
| 2019 | 25 | 0.56 |
| 2020 | 25 | 0.52 |
| 2021 | 25 | 0.48 |
| 2022 | 25 | 0.52 |
| 2023 | 25 | 0.32 |
| 2024 | 24 | 0.62 |
| 2025 | 25 | 0.44 |

#### Ceiling 30000 characters

| year | documents | divergence |
|---|---|---|
| 2018 | 25 | 0.68 |
| 2019 | 25 | 0.16 |
| 2020 | 25 | 0.88 |
| 2021 | 25 | 0.76 |
| 2022 | 25 | 0.64 |
| 2023 | 25 | 0.44 |
| 2024 | 24 | 0.62 |
| 2025 | 25 | 0.16 |

#### Ceiling 40000 characters

| year | documents | divergence |
|---|---|---|
| 2018 | 25 | 0.52 |
| 2019 | 25 | 0.48 |
| 2020 | 25 | 0.64 |
| 2021 | 25 | 0.80 |
| 2022 | 25 | 0.48 |
| 2023 | 25 | 0.60 |
| 2024 | 24 | 0.92 |
| 2025 | 25 | 0.48 |

### Concentration of the divergences

| cap | documents | documents with 0 divergence | docs holding 50% of divergence | share of those docs (%) | total divergence |
|---|---|---|---|---|---|
| 20000 | 199 | 152 | 16 | 51.69 | 89 |
| 30000 | 199 | 151 | 13 | 51.85 | 108 |
| 40000 | 199 | 133 | 20 | 50.00 | 122 |

## 10. Latency (indicative only)

Wall-clock times depend on the provider and on concurrent load; they are reported for completeness, not as a decision criterion.

### Ceiling 20000 characters

| arm | context (s) | extraction (s) |
|---|---|---|
| full | 0.00 | 5.29 |
| structured | 3.95 | 5.97 |
| flat_legacy | 1.41 | 6.71 |
| flat_v2 | 0.79 | 5.30 |

### Ceiling 30000 characters

| arm | context (s) | extraction (s) |
|---|---|---|
| full | 0.00 | 7.47 |
| structured | 4.52 | 7.50 |
| flat_legacy | 0.60 | 6.77 |
| flat_v2 | 0.54 | 7.35 |

### Ceiling 40000 characters

| arm | context (s) | extraction (s) |
|---|---|---|
| full | 0.00 | 10.05 |
| structured | 5.69 | 8.15 |
| flat_legacy | 0.60 | 8.40 |
| flat_v2 | 0.54 | 9.01 |

## 11. Findings

1. **20k**: `structured` sends 84 % of the characters of `full` but costs 115 % of it in tokens (2.92 vs 2.55 USD per 1000 documents); the outline selector alone is 17 % of its input tokens.
2. **30k**: `structured` sends 80 % of the characters of `full` but costs 113 % of it in tokens (3.27 vs 2.90 USD per 1000 documents); the outline selector alone is 17 % of its input tokens.
3. **40k**: `structured` sends 77 % of the characters of `full` but costs 109 % of it in tokens (3.57 vs 3.29 USD per 1000 documents); the outline selector alone is 18 % of its input tokens.
4. **20k**: `structured` reproduces 53 % of its mentions identically, 24 % with other labels only, and covers 83 % of what `full` found; 63 % of documents are strictly identical.
5. **30k**: `structured` reproduces 49 % of its mentions identically, 25 % with other labels only, and covers 82 % of what `full` found; 62 % of documents are strictly identical.
6. **40k**: `structured` reproduces 42 % of its mentions identically, 34 % with other labels only, and covers 80 % of what `full` found; 44 % of documents are strictly identical.
7. **20k**, rate-bearing rules only: `structured` covers 89 % of the rules of `full` (55 % identical, 12 % with other bounds) — versus 83 % for `flat_v2` and 72 % for `flat_legacy`.
8. **20k**: the new chunks (`flat_v2`) cover 83 % of the mentions of `full` against 72 % for the legacy collection (`flat_legacy`), i.e. +10 points.
9. **30k**, rate-bearing rules only: `structured` covers 82 % of the rules of `full` (50 % identical, 3 % with other bounds) — versus 66 % for `flat_v2` and 66 % for `flat_legacy`.
10. **30k**: the new chunks (`flat_v2`) cover 66 % of the mentions of `full` against 66 % for the legacy collection (`flat_legacy`), i.e. +0 points.
11. **40k**, rate-bearing rules only: `structured` covers 86 % of the rules of `full` (40 % identical, 9 % with other bounds) — versus 66 % for `flat_v2` and 52 % for `flat_legacy`.
12. **40k**: the new chunks (`flat_v2`) cover 66 % of the mentions of `full` against 52 % for the legacy collection (`flat_legacy`), i.e. +14 points.
13. **20k**: 54 % of the divergences of `structured` concern bare reminders without any rate; only 41 of 89 involve a rate-bearing rule.
14. **30k**: 51 % of the divergences of `structured` concern bare reminders without any rate; only 53 of 108 involve a rate-bearing rule.
15. **40k**: 48 % of the divergences of `structured` concern bare reminders without any rate; only 63 of 122 involve a rate-bearing rule.
16. **20k**: fabrication is rare and not specific to an arm — 2.2 to 4.0 % of citations contain no verbatim run of ten words (worst: flat_legacy), while 86–92 % are verbatim.
17. **30k**: fabrication is rare and not specific to an arm — 4.1 to 6.9 % of citations contain no verbatim run of ten words (worst: flat_legacy), while 81–89 % are verbatim.
18. **40k**: fabrication is rare and not specific to an arm — 1.1 to 6.7 % of citations contain no verbatim run of ten words (worst: flat_legacy), while 83–93 % are verbatim.
19. Divergence is concentrated: at 30k, 13 documents out of 199 hold half of it, and 151 documents have none.
20. Concentration grows with the ceiling: 20 documents hold half the divergence at 40k.
21. Mean divergence per document with the ceiling: 20k = 0.45, 30k = 0.54, 40k = 0.61.

## Reading the numbers

- `full` is the reference, not truth; a divergence means the two contexts led to different extractions, not that `full` is right.
- Yield and agreement must be read together: an arm can match `full` perfectly while missing the same rule (`full` is wrong for both), and an arm can be cheaper and more complete at the same time.
- Costs are token-based and exclude the embedding calls of the flat arms and the retrieval latency; the selector of `structured` is included.
- Latency is reported for completeness only: it depends on the provider and on concurrent load, not on the context strategy.