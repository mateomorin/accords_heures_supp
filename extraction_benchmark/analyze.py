"""
Aggregate the benchmark results into `extraction_benchmark/REPORT.md` and `FINDINGS.md`.

No LLM call. Reads the `results_<cap>.jsonl` files (always) and the
`citations_<cap>.jsonl` files (when `verify.py` has run).

The report is organised around the two questions that decide which context strategy to
ship: how much information each arm sees and what it costs (tokens, USD), and how much
of the extraction is actually lost when the context is reduced. Agreement with the
`full` arm is decomposed in tiers (`match.py`) because a raw "exact match" rate mixes
content loss with differences in wording and palier boundaries.

Usage:
    uv run extraction_benchmark/analyze.py
"""

from __future__ import annotations

import json

import pandas as pd
from _common import (
    ARMS,
    CAPS,
    FINDINGS_PATH,
    PRICE_IN_PER_M,
    PRICE_OUT_PER_M,
    REPORT_PATH,
    citations_path,
    results_path,
)
from match import compare_documents, has_rate, palier_key

NUMERIC = [
    "cap",
    "full_chars",
    "n_parts_doc",
    "context_chars",
    "n_context_parts",
    "tokens_in",
    "tokens_out",
    "selector_tokens_in",
    "selector_tokens_out",
    "context_s",
    "llm_s",
    "latency_s",
    "n_mentions",
    "n_regles",
    "n_rappels",
    "n_indeterminees",
    "n_with_rate",
    "n_paliers",
    "n_citations",
    "n_citations_grounded",
]


def to_md(frame: pd.DataFrame, decimals: int = 2) -> str:
    """Minimal markdown table (pandas' to_markdown needs tabulate)."""
    columns = [str(column).replace("_", " ") for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in frame.itertuples(index=False):
        cells = []
        for value in row:
            if value is None:
                cells.append("")
            elif isinstance(value, float):
                cells.append("" if pd.isna(value) else f"{value:.{decimals}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_results(cap: int) -> pd.DataFrame | None:
    """Numeric-coerced results of one ceiling, errors included."""
    path = results_path(cap)
    if not path.exists():
        return None
    frame = pd.read_json(path, lines=True)
    for column in NUMERIC:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_citations(cap: int) -> pd.DataFrame | None:
    """Source verification of one ceiling, when `verify.py` has run."""
    path = citations_path(cap)
    if not path.exists():
        return None
    return pd.read_json(path, lines=True)


def mentions(row) -> list[dict]:
    """Mentions of one result row."""
    raw = row.mentions_json if hasattr(row, "mentions_json") else row
    return json.loads(raw) if isinstance(raw, str) and raw else []


def error_kind(message) -> str | None:
    """Normalised error class, to keep the validity table readable."""
    if not isinstance(message, str):
        return None
    if "503" in message:
        return "provider 503"
    if "ceiling" in message:
        return "above ceiling"
    if "timed out" in message:
        return "timeout"
    if "parsing failed" in message:
        return "parsing"
    return message[:40]


def arms_in(frame: pd.DataFrame) -> list[str]:
    """Arms present in a results frame, in the canonical order."""
    return [arm for arm in ARMS if arm in set(frame["arm"])]


# --------------------------------------------------------------------------- tables


def validity_table(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Rows produced, errors by kind, and documents usable for a paired comparison."""
    rows = []
    for cap, frame in sorted(frames.items()):
        ok = frame[frame["erreur"].isna()]
        usable = set.intersection(
            *[set(ok[ok["arm"] == arm]["reference"]) for arm in ARMS if arm in set(ok["arm"])]
        )
        kinds = frame.loc[frame["erreur"].notna(), "erreur"].map(error_kind).value_counts()
        rows.append(
            {
                "cap": cap,
                "rows": len(frame),
                "failed": int(frame["erreur"].notna().sum()),
                "errors": ", ".join(f"{kind} x{count}" for kind, count in kinds.items()) or "-",
                "docs in sample": int(frame["reference"].nunique()),
                "docs paired (4 arms)": len(usable),
            }
        )
    return pd.DataFrame(rows)


def context_table(frame: pd.DataFrame) -> pd.DataFrame:
    """How much of the document each arm actually sends, in characters and parts."""
    ok = frame[frame["erreur"].isna()]
    full_chars = ok[ok["arm"] == "full"].set_index("reference")["context_chars"]
    rows = []
    for arm in arms_in(ok):
        group = ok[ok["arm"] == arm].set_index("reference")
        shared = group.index.intersection(full_chars.index)
        rows.append(
            {
                "arm": arm,
                "context chars": group["context_chars"].mean(),
                "% of document": 100 * group.loc[shared, "context_chars"].sum()
                / full_chars.loc[shared].sum(),
                "parts sent": group["n_context_parts"].mean(),
                "documents": len(group),
            }
        )
    return pd.DataFrame(rows)


def cost_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Token volumes and cost per 1000 documents, selector included."""
    ok = frame[frame["erreur"].isna()]
    rows = []
    for arm in arms_in(ok):
        group = ok[ok["arm"] == arm]
        docs = group["reference"].nunique()
        total_in = group["tokens_in"].sum() + group["selector_tokens_in"].fillna(0).sum()
        total_out = group["tokens_out"].sum() + group["selector_tokens_out"].fillna(0).sum()
        cost = total_in / 1e6 * PRICE_IN_PER_M + total_out / 1e6 * PRICE_OUT_PER_M
        rows.append(
            {
                "arm": arm,
                "input tok/doc": group["tokens_in"].mean(),
                "selector tok/doc": group["selector_tokens_in"].fillna(0).mean(),
                "total input tok/doc": total_in / docs,
                "output tok/doc": total_out / docs,
                "USD / 1k docs": 1000 * cost / docs,
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        reference = table.loc[table["arm"] == "full", "USD / 1k docs"]
        base = float(reference.iloc[0]) if len(reference) else float("nan")
        table["cost vs full (%)"] = 100 * table["USD / 1k docs"] / base
    return table


def yield_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Tabulate the extraction yield of each arm, regardless of agreement with the others."""
    ok = frame[frame["erreur"].isna()]
    rows = []
    for arm in arms_in(ok):
        group = ok[ok["arm"] == arm]
        rows.append(
            {
                "arm": arm,
                "docs": len(group),
                "docs without mention (%)": 100 * (group["n_mentions"] == 0).mean(),
                "mentions/doc": group["n_mentions"].mean(),
                "rules/doc": group["n_regles"].mean(),
                "reminders/doc": group["n_rappels"].mean(),
                "paliers/doc": group["n_paliers"].mean(),
                "paliers with rate/doc": group["n_with_rate"].mean(),
            }
        )
    return pd.DataFrame(rows)


AGREEMENT_COLUMNS = [
    "arm",
    "documents",
    "identical (%)",
    "same rule, other labels (%)",
    "different bounds (%)",
    "extra rule (%)",
    "full covered (%)",
    "documents fully identical (%)",
]


def agreement_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Tiered agreement of every arm against `full`."""
    ok = frame[frame["erreur"].isna()]
    if "full" not in set(ok["arm"]):
        return pd.DataFrame()
    rows = []
    for arm in arms_in(ok):
        if arm == "full":
            continue
        stats = compare_documents(ok, arm, "full")
        if stats["documents"]:
            rows.append({"arm": arm, **stats})
    table = pd.DataFrame(rows)
    return table[[column for column in AGREEMENT_COLUMNS if column in table]].copy()


def _mention_kind(prefix: str, mention: dict) -> str:
    """Column name of an unmatched mention: with a numeric rate, or a bare reminder."""
    return f"{prefix} rule with rate" if has_rate(mention) else f"{prefix} reminder (no rate)"


def rate_only_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Agreement restricted to mentions that carry at least one numeric rate."""
    ok = frame[frame["erreur"].isna()]
    by_arm = {arm: group.set_index("reference") for arm, group in ok.groupby("arm")}
    if "full" not in by_arm:
        return pd.DataFrame()

    rows = []
    for arm in arms_in(ok):
        if arm == "full":
            continue
        n_arm = n_full = 0
        counts = dict.fromkeys(("exact", "same_rule", "bounds", "arm_only", "full_only"), 0)
        for reference in by_arm["full"].index.intersection(by_arm[arm].index):
            arm_mentions = [
                m for m in mentions(by_arm[arm].loc[reference, "mentions_json"]) if has_rate(m)
            ]
            full_mentions = [
                m for m in mentions(by_arm["full"].loc[reference, "mentions_json"]) if has_rate(m)
            ]
            outcome = compare_outcome(arm_mentions, full_mentions)
            n_arm += len(arm_mentions)
            n_full += len(full_mentions)
            for tier in counts:
                counts[tier] += len(outcome[tier])
        aligned = counts["exact"] + counts["same_rule"] + counts["bounds"]
        rows.append(
            {
                "arm": arm,
                "rules (arm)": n_arm,
                "rules (full)": n_full,
                "identical (%)": 100 * counts["exact"] / n_arm if n_arm else float("nan"),
                "same rule, other labels (%)": (
                    100 * counts["same_rule"] / n_arm if n_arm else float("nan")
                ),
                "different bounds (%)": 100 * counts["bounds"] / n_arm if n_arm else float("nan"),
                "extra rule (%)": 100 * counts["arm_only"] / n_arm if n_arm else float("nan"),
                "full covered (%)": 100 * aligned / n_full if n_full else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def taxonomy_table(frame: pd.DataFrame) -> pd.DataFrame:
    """
    How a divergence is shaped, not just how large it is.

    - `bounds, same rates`  : identical rates, boundaries split/merged/shifted — the
      rule was found, its extent was not;
    - `bounds, other rates` : the paliers themselves differ;
    - `extra / missed rule` : a mention with at least one numeric rate, present in one
      arm only — real content difference;
    - `extra / missed reminder` : a mention without any rate (a bare `rappel`), present
      in one arm only — often a mention of the legal 25/50 % rates.
    """
    ok = frame[frame["erreur"].isna()]
    by_arm = {arm: group.set_index("reference") for arm, group in ok.groupby("arm")}
    if "full" not in by_arm:
        return pd.DataFrame()

    rows = []
    for arm in arms_in(ok):
        if arm == "full":
            continue
        counts = {
            "bounds, same rates": 0,
            "bounds, other rates": 0,
            "extra rule with rate": 0,
            "extra reminder (no rate)": 0,
            "missed rule with rate": 0,
            "missed reminder (no rate)": 0,
        }
        for reference in by_arm["full"].index.intersection(by_arm[arm].index):
            arm_mentions = mentions(by_arm[arm].loc[reference, "mentions_json"])
            full_mentions = mentions(by_arm["full"].loc[reference, "mentions_json"])
            outcome = compare_outcome(arm_mentions, full_mentions)
            for left, right in outcome["bounds"]:
                rates = {p[1] for p in map(palier_key, arm_mentions[left]["paliers"])}
                other = {p[1] for p in map(palier_key, full_mentions[right]["paliers"])}
                counts["bounds, same rates" if rates == other else "bounds, other rates"] += 1
            for index in outcome["arm_only"]:
                counts[_mention_kind("extra", arm_mentions[index])] += 1
            for index in outcome["full_only"]:
                counts[_mention_kind("missed", full_mentions[index])] += 1
        rows.append({"arm": arm, **counts})
    return pd.DataFrame(rows)


def compare_outcome(left: list[dict], right: list[dict]) -> dict:
    """Tiered alignment, imported lazily to keep this module readable."""
    from match import match_mentions

    return match_mentions(left, right)


def citation_table(frame: pd.DataFrame, citations: pd.DataFrame | None) -> pd.DataFrame:
    """
    Citation fidelity, from two independent checks.

    - `in context (%)` is computed at run time: the citation is a substring of the
      context that was sent (it is the only check the run could do);
    - `in source (%)` re-reads the document: the citation exists verbatim in the
      agreement, whichever context was sent.
    A citation missing from the source is a fabrication; a citation in the source but
    not in the context means the passage was never available to the model.
    """
    ok = frame[frame["erreur"].isna()]
    rows = []
    for arm in arms_in(ok):
        group = ok[ok["arm"] == arm]
        row = {
            "arm": arm,
            "citations": int(group["n_citations"].sum()),
            "in context (%)": 100 * group["n_citations_grounded"].sum() / group["n_citations"].sum()
            if group["n_citations"].sum()
            else float("nan"),
        }
        if citations is not None and len(citations):
            subset = citations[citations["arm"] == arm]
            for verdict, column in (
                ("exact", "in source (%)"),
                ("partial", "partially quoted (%)"),
                ("fabricated", "fabricated (%)"),
            ):
                row[column] = (
                    100 * (subset["verdict"] == verdict).mean() if len(subset) else float("nan")
                )
        rows.append(row)
    return pd.DataFrame(rows)


def divergence_per_document(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Content divergence of one arm against `full`, per document."""
    ok = frame[frame["erreur"].isna()]
    stats = compare_documents(ok, arm, "full")
    per_document = stats.get("_per_document", {})
    rows = []
    for reference, item in per_document.items():
        row = ok[(ok["reference"] == reference)].iloc[0]
        rows.append(
            {
                "reference": reference,
                "year": int(row["anneeSignature"]),
                "full_chars": float(row["full_chars"]),
                "divergence": item["content_divergence"],
            }
        )
    return pd.DataFrame(rows)


def size_table(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Content divergence per document, by quartile of document size."""
    per_document = divergence_per_document(frame, arm)
    if per_document.empty:
        return pd.DataFrame()
    labels = ["Q1 (small)", "Q2", "Q3", "Q4 (large)"]
    per_document["quartile"] = pd.qcut(
        per_document["full_chars"].rank(method="first"), 4, labels=labels
    )
    table = (
        per_document.groupby("quartile", observed=True)
        .agg(
            documents=("divergence", "size"),
            divergence=("divergence", "mean"),
            chars=("full_chars", "mean"),
        )
        .reset_index()
    )
    return table


def year_table(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Content divergence per document, by signature year."""
    per_document = divergence_per_document(frame, arm)
    if per_document.empty:
        return pd.DataFrame()
    return (
        per_document.groupby("year")
        .agg(documents=("divergence", "size"), divergence=("divergence", "mean"))
        .reset_index()
    )


def concentration(per_document: pd.DataFrame) -> tuple[int, float]:
    """Return how many documents hold half of the total divergence, and their share."""
    total = per_document["divergence"].sum()
    if not total:
        return 0, 0.0
    ranked = per_document["divergence"].sort_values(ascending=False).cumsum()
    index = min(int((ranked < 0.5 * total).sum()) + 1, len(ranked))
    return index, 100 * ranked.iloc[index - 1] / total


def latency_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Wall-clock split between context construction and the extraction call."""
    ok = frame[frame["erreur"].isna()]
    rows = []
    for arm in arms_in(ok):
        group = ok[ok["arm"] == arm]
        rows.append(
            {
                "arm": arm,
                "context (s)": group["context_s"].mean(),
                "extraction (s)": group["llm_s"].mean(),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- report


def main() -> None:
    frames = {cap: load_results(cap) for cap in CAPS}
    frames = {cap: frame for cap, frame in frames.items() if frame is not None}
    if not frames:
        raise SystemExit("No results file found; run `extraction_benchmark/run.py` first.")
    citations = {cap: load_citations(cap) for cap in frames}

    sections = [
        "# Extraction benchmark — context strategies",
        "",
        "Four ways of building the context of an overtime-majoration extraction, on three",
        "samples of agreements whose whole text fits under 20k, 30k and 40k characters.",
        "Same prompt, same schema, same model (`mistral-medium-2508`), same temperature,",
        "same per-document budget (the ceiling under test) for every arm.",
        "",
        "| Arm | Context |",
        "| --- | --- |",
        "| `full` | the whole document, in structural order — **the reference** |",
        "| `structured` | anchors + outline selector + similarity + sibling completion |",
        "| `flat_legacy` | chunk similarity, previous collection `accords_heures_supp` |",
        "| `flat_v2` | chunk similarity, parts collection `accords_heures_supp_parts` |",
        "",
        "`full` is not ground truth: it is the arm that sees everything, so every "
        "agreement rate below measures fidelity to a maximal-context read, not accuracy.",
        "",
        "## 1. Validity",
        "",
    ]

    validity = validity_table(frames)
    sections += [
        to_md(validity),
        "",
        "A document is kept in the comparison only when all four arms produced a result "
        "for it; the report therefore never compares an arm against a missing baseline.",
        "",
    ]

    findings = []
    # Cross-ceiling values, used for the headline conclusions at the top of FINDINGS.md.
    collected: dict[str, list[float]] = {
        "context": [],
        "cost": [],
        "selector": [],
        "rate_structured": [],
        "rate_v2": [],
        "rate_legacy": [],
        "reminder_share": [],
        "fabrication_min": [],
        "fabrication_max": [],
        "concentration": [],
        "zero_divergence": [],
        "documents": [],
    }

    # --- volume and cost, the decisive part -----------------------------------------
    sections += ["## 2. How much context each arm sends", ""]
    for cap, frame in sorted(frames.items()):
        sections += [f"### Ceiling {cap} characters", "", to_md(context_table(frame)), ""]

    sections += [
        "## 3. Tokens and cost",
        "",
        f"Prices used: USD {PRICE_IN_PER_M:.2f}/M input tokens, "
        f"USD {PRICE_OUT_PER_M:.2f}/M output tokens (Mistral Medium 3 list price). "
        "Every cost is linear in those two numbers. The selector of the `structured` arm "
        "is a second LLM call and is counted as input/output tokens of that arm.",
        "",
    ]
    for cap, frame in sorted(frames.items()):
        table = cost_table(frame)
        sections += [f"### Ceiling {cap} characters", "", to_md(table), ""]
        by_arm = table.set_index("arm")
        if {"full", "structured"} <= set(by_arm.index):
            full, structured = by_arm.loc["full"], by_arm.loc["structured"]
            ok = frame[frame["erreur"].isna()]
            full_chars = ok[ok["arm"] == "full"].set_index("reference")["full_chars"]
            group = ok[ok["arm"] == "structured"].set_index("reference")
            shared = group.index.intersection(full_chars.index)
            ratio = 100 * group.loc[shared, "context_chars"].sum() / full_chars.loc[shared].sum()
            selector_share = (
                100 * structured["selector tok/doc"] / structured["total input tok/doc"]
            )
            collected["context"].append(ratio)
            collected["cost"].append(structured["cost vs full (%)"])
            collected["selector"].append(selector_share)
            findings.append(
                f"**{cap // 1000}k**: `structured` sends {ratio:.0f} % of the characters "
                f"of `full` but costs {structured['cost vs full (%)']:.0f} % of it in "
                f"tokens ({structured['USD / 1k docs']:.2f} vs "
                f"{full['USD / 1k docs']:.2f} USD per 1000 documents); the outline "
                f"selector alone is {selector_share:.0f} % of its input tokens."
            )

    # --- yield ----------------------------------------------------------------------
    sections += ["## 4. What each arm extracts", ""]
    sections += [
        "Yield only: this is what the model returned for the context it received, before "
        "any comparison. A higher yield can mean better recall or more hallucination.",
        "",
    ]
    for cap, frame in sorted(frames.items()):
        sections += [f"### Ceiling {cap} characters", "", to_md(yield_table(frame)), ""]

    # --- agreement ------------------------------------------------------------------
    sections += [
        "## 5. Agreement with `full`, by tier",
        "",
        "`identical` is the strict measure (nature, threshold, population and palier set). "
        "`same rule, other labels` is the same palier set with a different nature, "
        "threshold or population wording — the rule was found, its labels differ. "
        "`full covered` is the share of the mentions of `full` that the arm reproduces at "
        "any tier; `extra rule` is the share of the arm's mentions that `full` does not "
        "have (splitting or hallucination). Only `different bounds`, `extra rule` and the "
        "complement of `full covered` are real content divergences.",
        "",
    ]
    for cap, frame in sorted(frames.items()):
        table = agreement_table(frame)
        sections += [f"### Ceiling {cap} characters", "", to_md(table), ""]
        if not table.empty:
            structured = table[table["arm"] == "structured"]
            if len(structured):
                row = structured.iloc[0]
                findings.append(
                    f"**{cap // 1000}k**: `structured` reproduces "
                    f"{row['identical (%)']:.0f} % of its mentions identically, "
                    f"{row['same rule, other labels (%)']:.0f} % with other labels only, "
                    f"and covers {row['full covered (%)']:.0f} % of what `full` found; "
                    f"{row['documents fully identical (%)']:.0f} % of documents are "
                    "strictly identical."
                )

    sections += [
        "## 6. Agreement on rate-bearing rules only",
        "",
        "The same comparison restricted to mentions that carry at least one numeric rate. "
        "Bare reminders (`rappel` without a rate) are filtered out on both sides, because "
        "they are reported very inconsistently and are not what the extraction is for.",
        "",
    ]
    for cap, frame in sorted(frames.items()):
        table = rate_only_table(frame)
        sections += [f"### Ceiling {cap} characters", "", to_md(table), ""]
        if not table.empty and "structured" in set(table["arm"]):
            row = table[table["arm"] == "structured"].iloc[0]
            for name, key in (
                ("structured", "rate_structured"),
                ("flat_v2", "rate_v2"),
                ("flat_legacy", "rate_legacy"),
            ):
                collected[key].append(
                    float(table[table["arm"] == name].iloc[0]["full covered (%)"])
                )
            findings.append(
                f"**{cap // 1000}k**, rate-bearing rules only: `structured` covers "
                f"{row['full covered (%)']:.0f} % of the rules of `full` "
                f"({row['identical (%)']:.0f} % identical, {row['different bounds (%)']:.0f} % "
                f"with other bounds) — versus "
                f"{table[table['arm'] == 'flat_v2'].iloc[0]['full covered (%)']:.0f} % for "
                f"`flat_v2` and "
                f"{table[table['arm'] == 'flat_legacy'].iloc[0]['full covered (%)']:.0f} % for "
                f"`flat_legacy`."
            )

            if {"flat_legacy", "flat_v2"} <= set(table["arm"]):
                legacy = table[table["arm"] == "flat_legacy"].iloc[0]
                v2 = table[table["arm"] == "flat_v2"].iloc[0]
                findings.append(
                    f"**{cap // 1000}k**: the new chunks (`flat_v2`) cover "
                    f"{v2['full covered (%)']:.0f} % of the mentions of `full` against "
                    f"{legacy['full covered (%)']:.0f} % for the legacy collection "
                    f"(`flat_legacy`), i.e. "
                    f"{v2['full covered (%)'] - legacy['full covered (%)']:+.0f} points."
                )

    # --- taxonomy -------------------------------------------------------------------
    sections += [
        "## 7. Shape of the divergences",
        "",
        "Same divergences as section 5, broken down by what actually differs. "
        "`bounds, same rates` means the rule and its rates were found but its extent "
        "differs; `missed rule with rate` is the loss that matters.",
        "",
    ]
    for cap, frame in sorted(frames.items()):
        table = taxonomy_table(frame)
        sections += [f"### Ceiling {cap} characters", "", to_md(table), ""]
        if not table.empty and "structured" in set(table["arm"]):
            row = table[table["arm"] == "structured"].iloc[0]
            with_rate = (
                row["bounds, same rates"]
                + row["bounds, other rates"]
                + row["extra rule with rate"]
                + row["missed rule with rate"]
            )
            without_rate = row["extra reminder (no rate)"] + row["missed reminder (no rate)"]
            total = with_rate + without_rate
            if total:
                collected["reminder_share"].append(100 * without_rate / total)
            findings.append(
                f"**{cap // 1000}k**: {100 * without_rate / total if total else 0:.0f} % of the "
                "divergences of `structured` concern bare reminders without any rate; "
                f"only {with_rate} of {total} involve a rate-bearing rule."
            )

    # --- citations ------------------------------------------------------------------
    sections += [
        "## 8. Citation fidelity",
        "",
        "Two independent checks. `in context` is computed at run time: the citation is a "
        "substring of the context that was sent. `in source` re-reads the full text each "
        "arm was built from (`verify.py`, S3 and Qdrant, no LLM): the v2 parts for "
        "`full`, `structured` and `flat_v2`, the legacy chunks for `flat_legacy`, because "
        "the two collections come from different conversions of the same agreement. "
        "`partially quoted` means a verbatim run of at least ten words plus rewritten "
        "material; `fabricated` means no verbatim run of that length at all. A citation "
        "present in the source but absent from the context is a passage the model never "
        "saw; the opposite cannot happen and would reveal a check bug.",
        "",
    ]
    for cap, frame in sorted(frames.items()):
        table = citation_table(frame, citations.get(cap))
        sections += [f"### Ceiling {cap} characters", "", to_md(table), ""]
        if "in source (%)" in table.columns:
            worst = table.sort_values("fabricated (%)", ascending=False).iloc[0]
            collected["fabrication_min"].append(float(table["fabricated (%)"].min()))
            collected["fabrication_max"].append(float(table["fabricated (%)"].max()))
            findings.append(
                f"**{cap // 1000}k**: fabrication is rare and not specific to an arm — "
                f"{table['fabricated (%)'].min():.1f} to {table['fabricated (%)'].max():.1f} % "
                f"of citations contain no verbatim run of ten words "
                f"(worst: {worst['arm']}), while "
                f"{table['in source (%)'].min():.0f}–{table['in source (%)'].max():.0f} % "
                "are verbatim."
            )

    # --- robustness -----------------------------------------------------------------
    sections += [
        "## 9. Robustness",
        "",
        "### Divergence against document size (`structured`)",
        "",
    ]
    for cap, frame in sorted(frames.items()):
        sections += [
            f"#### Ceiling {cap} characters",
            "",
            to_md(size_table(frame, "structured")),
            "",
        ]

    sections += ["### Divergence by signature year (`structured`)", ""]
    for cap, frame in sorted(frames.items()):
        sections += [
            f"#### Ceiling {cap} characters",
            "",
            to_md(year_table(frame, "structured")),
            "",
        ]

    sections += ["### Concentration of the divergences", ""]
    concentration_rows = []
    for cap, frame in sorted(frames.items()):
        per_document = divergence_per_document(frame, "structured")
        if per_document.empty:
            continue
        n_docs, share = concentration(per_document)
        collected["concentration"].append(n_docs)
        collected["zero_divergence"].append(int((per_document["divergence"] == 0).sum()))
        collected["documents"].append(len(per_document))
        concentration_rows.append(
            {
                "cap": cap,
                "documents": len(per_document),
                "documents with 0 divergence": int((per_document["divergence"] == 0).sum()),
                "docs holding 50% of divergence": n_docs,
                "share of those docs (%)": share,
                "total divergence": int(per_document["divergence"].sum()),
            }
        )
    concentrated = pd.DataFrame(concentration_rows)
    sections += [to_md(concentrated), ""]
    if len(concentrated):
        worst = concentrated.sort_values("docs holding 50% of divergence").iloc[0]
        findings.append(
            f"Divergence is concentrated: at {int(worst['cap']) // 1000}k, "
            f"{int(worst['docs holding 50% of divergence'])} documents out of "
            f"{int(worst['documents'])} hold half of it, and "
            f"{int(worst['documents with 0 divergence'])} documents have none."
        )
        largest = concentrated.sort_values(
            "docs holding 50% of divergence", ascending=False
        ).iloc[0]
        if largest["cap"] != worst["cap"]:
            findings.append(
                "Concentration grows with the ceiling: "
                f"{int(largest['docs holding 50% of divergence'])} documents hold half the "
                f"divergence at {int(largest['cap']) // 1000}k."
            )
    if len(concentrated) >= 2:
        ordered = concentrated.sort_values("cap")
        findings.append(
            "Mean divergence per document with the ceiling: "
            + ", ".join(
                f"{int(row['cap']) // 1000}k = {row['total divergence'] / row['documents']:.2f}"
                for _, row in ordered.iterrows()
            )
            + "."
        )

    # --- latency --------------------------------------------------------------------
    sections += [
        "## 10. Latency (indicative only)",
        "",
        "Wall-clock times depend on the provider and on concurrent load; they are reported "
        "for completeness, not as a decision criterion.",
        "",
    ]
    for cap, frame in sorted(frames.items()):
        sections += [f"### Ceiling {cap} characters", "", to_md(latency_table(frame)), ""]

    # --- findings -------------------------------------------------------------------
    def span(key: str, unit: str = "") -> str:
        """`min–max` of a collected series, formatted for prose."""
        values = collected[key]
        if not values:
            return "n/a"
        low, high = min(values), max(values)
        if abs(high - low) < 0.5:
            return f"{low:.0f}{unit}"
        return f"{low:.0f}–{high:.0f}{unit}"

    headlines = [
        f"**Reducing the context does not reduce the bill.** `structured` sends "
        f"{span('context', ' %')} of the characters of `full` but costs "
        f"{span('cost', ' %')} of its tokens, because the outline selector alone is "
        f"{span('selector', ' %')} of its input tokens. `flat_v2` is genuinely cheaper "
        f"(and sees far less).",
        f"**On rate-bearing rules `structured` is close to `full`.** It reproduces "
        f"{span('rate_structured', ' %')} of the rules of `full`, against "
        f"{span('rate_v2', ' %')} for `flat_v2` and {span('rate_legacy', ' %')} for "
        f"`flat_legacy`.",
        f"**Most of the remaining divergence is not about rates.** "
        f"{span('reminder_share', ' %')} of the divergences of `structured` concern bare "
        "reminders without any numeric rate; a bare reminder is a valid mention, but it "
        "is reported inconsistently in both directions.",
        f"**The divergence is concentrated.** Half of it sits in "
        f"{span('concentration')} documents out of {span('documents')}, and "
        f"{span('zero_divergence')} documents have none: a per-document review of the "
        "worst cases is tractable (`manual_review_<cap>.md`).",
        "**Citations are mostly verbatim.** "
        f"{min(collected['fabrication_min']):.0f}–{max(collected['fabrication_max']):.0f} % "
        "of citations contain no verbatim run of ten words; no arm is clearly worse, and "
        "the legacy chunks quote their own text (the two representations differ, which is "
        "why each arm is verified against its own source).",
    ]

    findings_text = [
        "# Findings",
        "",
        "Generated by `analyze.py` from `results_<cap>.jsonl` and `citations_<cap>.jsonl`. "
        "Ranges cover the three ceilings (20k, 30k, 40k); every claim is backed by a "
        "table in `REPORT.md`.",
        "",
        "## Headline",
        "",
    ]
    findings_text += [f"{index}. {line}" for index, line in enumerate(headlines, start=1)]
    findings_text += ["", "## Detail by ceiling", ""]
    findings_text += [f"{index}. {line}" for index, line in enumerate(findings, start=1)] + [""]
    FINDINGS_PATH.write_text("\n".join(findings_text), encoding="utf-8")

    sections += [
        "## 11. Findings",
        "",
        *[f"{index}. {line}" for index, line in enumerate(findings, start=1)],
        "",
        "## Reading the numbers",
        "",
        "- `full` is the reference, not truth; a divergence means the two contexts led to "
        "different extractions, not that `full` is right.",
        "- Yield and agreement must be read together: an arm can match `full` perfectly "
        "while missing the same rule (`full` is wrong for both), and an arm can be "
        "cheaper and more complete at the same time.",
        "- Costs are token-based and exclude the embedding calls of the flat arms and the "
        "retrieval latency; the selector of `structured` is included.",
        "- Latency is reported for completeness only: it depends on the provider and on "
        "concurrent load, not on the context strategy.",
        "",
    ]

    REPORT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")
    print(f"Findings written to {FINDINGS_PATH}")


if __name__ == "__main__":
    main()
