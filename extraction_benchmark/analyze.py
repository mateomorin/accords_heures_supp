"""
Aggregate the benchmark results and write `extraction_benchmark/REPORT.md`.

No LLM call, no network: reads the `results_<cap>.jsonl` files only.

Usage:
    uv run extraction_benchmark/analyze.py
"""

from __future__ import annotations

import pandas as pd
from _common import ARMS, CAPS, REPORT_PATH, results_path
from match import agreement_stats

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


def to_md(frame: pd.DataFrame) -> str:
    """Minimal markdown table (pandas' to_markdown needs tabulate)."""
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in frame.itertuples(index=False):
        cells = []
        for value in row:
            if value is None:
                cells.append("")
            elif isinstance(value, float):
                cells.append("" if pd.isna(value) else f"{value:.2f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ratio(frame: pd.DataFrame, numerator: str, denominator: str) -> float:
    total = frame[denominator].sum()
    return float(frame[numerator].sum() / total) if total else float("nan")


def load_results(cap: int) -> pd.DataFrame | None:
    path = results_path(cap)
    if not path.exists():
        return None
    frame = pd.read_json(path, lines=True)
    for column in NUMERIC:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def overall_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in [arm for arm in ARMS if arm in set(frame["arm"])]:
        group = frame[frame["arm"] == arm]
        ok = group[group["erreur"].isna()]
        rows.append(
            {
                "arm": arm,
                "docs": ok["reference"].nunique(),
                "errors": int(group["erreur"].notna().sum()),
                "parse_or_run_errors (%)": 100 * group["erreur"].notna().mean(),
                "context_chars": ok["context_chars"].mean(),
                "tokens_in/doc": ok["tokens_in"].mean(),
                "selector_tokens_in/doc": ok["selector_tokens_in"].mean(),
                "tokens_out/doc": ok["tokens_out"].mean(),
                "context_s": ok["context_s"].mean(),
                "llm_s": ok["llm_s"].mean(),
                "mentions/doc": ok["n_mentions"].mean(),
                "docs_with_rule (%)": 100 * (ok.groupby("reference")["n_regles"].max() > 0).mean(),
                "rules/doc": ok["n_regles"].mean(),
                "paliers/doc": ok["n_paliers"].mean(),
                "rates/doc": ok["n_with_rate"].mean(),
                "citations_grounded (%)": 100
                * ratio(ok, "n_citations_grounded", "n_citations"),
            }
        )
    return pd.DataFrame(rows)


def per_year_table(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    ok = frame[frame["erreur"].isna()]
    table = ok.pivot_table(
        index="anneeSignature", columns="arm", values=value, aggfunc="mean"
    )
    return table.reset_index().rename(columns={"anneeSignature": "year"})


def agreement_table(frame: pd.DataFrame) -> pd.DataFrame:
    ok = frame[frame["erreur"].isna()]
    if "full" not in set(ok["arm"]):
        return pd.DataFrame()
    rows = []
    for arm in ARMS:
        if arm == "full" or arm not in set(ok["arm"]):
            continue
        stats = agreement_stats(ok, arm, baseline="full")
        if stats["documents"]:
            rows.append({"arm": arm, **stats})
    return pd.DataFrame(rows)


def summary_row(cap: int, frame: pd.DataFrame) -> list[dict]:
    ok = frame[frame["erreur"].isna()]
    rows = []
    for arm in ARMS:
        group = ok[ok["arm"] == arm]
        if group.empty:
            continue
        rows.append(
            {
                "cap": cap,
                "arm": arm,
                "docs": group["reference"].nunique(),
                "context_chars": group["context_chars"].mean(),
                "tokens_in/doc": group["tokens_in"].mean(),
                "mentions/doc": group["n_mentions"].mean(),
                "rules/doc": group["n_regles"].mean(),
                "paliers/doc": group["n_paliers"].mean(),
                "grounded (%)": 100 * ratio(group, "n_citations_grounded", "n_citations"),
            }
        )
    return rows


def observations(cap: int, overall: pd.DataFrame) -> list[str]:
    by_arm = overall.set_index("arm")
    lines = []
    if {"full", "structured"} <= set(by_arm.index):
        full, structured = by_arm.loc["full"], by_arm.loc["structured"]
        lines.append(
            f"Structured vs full: {structured['mentions/doc']:.2f} vs "
            f"{full['mentions/doc']:.2f} mentions/doc "
            f"({100 * (structured['mentions/doc'] / full['mentions/doc'] - 1):+.1f}%), "
            f"{structured['paliers/doc']:.2f} vs {full['paliers/doc']:.2f} paliers/doc, "
            f"{structured['tokens_in/doc']:.0f} vs {full['tokens_in/doc']:.0f} input tokens/doc."
        )
    for arm in ("flat_legacy", "flat_v2"):
        if arm in by_arm.index:
            flat = by_arm.loc[arm]
            lines.append(
                f"{arm}: {flat['mentions/doc']:.2f} mentions/doc, "
                f"{flat['paliers/doc']:.2f} paliers/doc, "
                f"{flat['tokens_in/doc']:.0f} input tokens/doc, "
                f"{flat['citations_grounded (%)']:.1f}% grounded citations."
            )
    if "structured" in by_arm.index:
        selector = by_arm.loc["structured"]["selector_tokens_in/doc"]
        if pd.notna(selector):
            lines.append(
                f"The outline selector costs an extra {selector:.0f} input tokens/doc "
                f"(included in `context_s`, not in `tokens_in`)."
            )
    return lines


def main() -> None:
    frames = {cap: load_results(cap) for cap in CAPS}
    frames = {cap: frame for cap, frame in frames.items() if frame is not None}
    if not frames:
        raise SystemExit("No results file found; run `extraction_benchmark/run.py` first.")

    summary = []
    sections = [
        "# Extraction benchmark — context strategies",
        "",
        "## Protocol",
        "",
        "- Three samples of 25 documents per signature year (2018–2025), each drawn from "
        "the documents whose rendered length is at most 20k, 30k and 40k characters "
        "(`sample.py`); each ceiling has its own independent sample.",
        "- Four arms, identical prompt / schema / model / temperature:",
        "  - `full`: the whole document in structural order;",
        "  - `structured`: the production hybrid pipeline, forced (anchors + outline "
        "selector + similarity + sibling completion);",
        "  - `flat_legacy`: chunk similarity on `accords_heures_supp`;",
        "  - `flat_v2`: chunk similarity on `accords_heures_supp_parts`.",
        "- Every arm is bounded by the ceiling under test: a full reading would have been "
        "allowed up to that size, so the structured and flat contexts get the same budget "
        "(chunks are never truncated, retrieval stops when the budget is reached).",
        "- Benchmark-only prompts: `section_ids` is removed from the prompt and from the "
        "schema, because the flat arms receive chunks without `[section_id]` headers and "
        "could not satisfy the requirement. Every other instruction is the production one.",
        "- Metrics: exact API token usage, context and LLM latency separately, mentions / "
        "rules / paliers per document, and verbatim-citation grounding (the citation is a "
        "substring of the context actually sent).",
        "- Mentions are compared with an order-insensitive matcher (`match.py`): exact "
        "match on nature/threshold/population/paliers first, then a partial match when "
        "only the paliers differ. What remains is written to `manual_review_<cap>.md`.",
        "- There is no ground truth: the report measures yield, grounding, agreement and "
        "cost, and lists the ambiguous cases for manual adjudication.",
        "",
    ]

    for cap, frame in sorted(frames.items()):
        overall = overall_table(frame)
        summary += summary_row(cap, frame)
        sections += [
            f"## Ceiling {cap} characters",
            "",
            f"{frame['reference'].nunique()} documents, {len(frame)} runs.",
            "",
            "### Overall",
            "",
            to_md(overall),
            "",
            "### Mentions per document, per year",
            "",
            to_md(per_year_table(frame, "n_mentions")),
            "",
            "### Conventional rules per document, per year",
            "",
            to_md(per_year_table(frame, "n_regles")),
            "",
            "### Agreement with the `full` arm",
            "",
            to_md(agreement_table(frame)),
            "",
            "### Observations",
            "",
            "\n".join(f"- {line}" for line in observations(cap, overall)),
            "",
        ]

    sections += [
        "## Summary across ceilings",
        "",
        to_md(pd.DataFrame(summary)),
        "",
        "## Reading the numbers",
        "",
        "- `mentions/doc`, `rules/doc` and `paliers/doc` are yields, not accuracy: a higher "
        "yield can mean better recall or more hallucination. The grounding rate and the "
        "manual review file are there to tell them apart.",
        "- The flat arms see far fewer characters than `full` by construction; "
        "`context_chars` and `tokens_in/doc` quantify that difference.",
        "- Restricting the sample below a ceiling makes every document readable in full, "
        "which is precisely the regime where the hybrid context has to prove it does not "
        "lose rules.",
        "",
    ]

    REPORT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
