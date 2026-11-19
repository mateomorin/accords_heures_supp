"""
Order-insensitive comparison of the mentions produced by the four arms.

The LLM may return the same rules in a different order, split or merge mentions, or
attach the same rate to a different population. This module canonicalises a mention and
aligns two lists of mentions in two tiers:

  1. exact   : same nature, same threshold, same population, same set of paliers;
  2. partial : same nature/threshold/population but a different set of paliers.

What cannot be aligned is left unmatched and written to `manual_review_<cap>.md` (one
section per ambiguous document, every arm side by side) so it can be adjudicated by hand.

Usage:
    uv run extraction_benchmark/match.py --caps 20000,30000,40000
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import pandas as pd
from _common import CAPS, results_path, review_path

MAX_REVIEW_DOCS = 40


def normalize(text: str | None) -> str:
    """Case, accent and whitespace insensitive form used to compare populations."""
    text = (text or "").replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return re.sub(r"[^a-z0-9' ]", "", text)


def number(value) -> float | None:
    """Return a rounded float, so 25, 25.0 and 25.00 compare equal."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def palier_key(palier: dict) -> tuple:
    """Order-insensitive identity of one palier."""
    return (
        palier.get("type_majoration"),
        number(palier.get("taux_valeur")),
        number(palier.get("rang_debut")),
        number(palier.get("rang_fin")),
        number(palier.get("volume_heures_debut")),
        number(palier.get("volume_heures_fin")),
    )


def mention_key(mention: dict) -> tuple:
    """Identity of a mention without its paliers."""
    return (
        mention.get("nature"),
        number(mention.get("seuil_declenchement")),
        normalize(mention.get("population_concernee")),
    )


def mention_signature(mention: dict) -> tuple:
    """Full, order-insensitive identity of a mention."""
    return mention_key(mention) + (tuple(sorted(palier_key(p) for p in mention["paliers"])),)


def match_mentions(left: list[dict], right: list[dict]) -> dict:
    """Align two mention lists: exact pairs, then partial pairs, then leftovers."""
    available: dict[tuple, list[int]] = defaultdict(list)
    for index, mention in enumerate(right):
        available[mention_signature(mention)].append(index)

    exact: list[tuple[int, int]] = []
    remaining_left = []
    for index, mention in enumerate(left):
        bucket = available.get(mention_signature(mention))
        if bucket:
            exact.append((index, bucket.pop(0)))
        else:
            remaining_left.append(index)

    used_right = {index for _, index in exact}
    remaining_right = [index for index in range(len(right)) if index not in used_right]

    by_key: dict[tuple, list[int]] = defaultdict(list)
    for index in remaining_right:
        by_key[mention_key(right[index])].append(index)

    partial: list[tuple[int, int]] = []
    unmatched_left = []
    for left_index in remaining_left:
        bucket = by_key.get(mention_key(left[left_index]))
        if bucket:
            partial.append((left_index, bucket.pop(0)))
        else:
            unmatched_left.append(left_index)

    matched_right = used_right | {index for _, index in partial}
    unmatched_right = [index for index in range(len(right)) if index not in matched_right]

    return {
        "exact": exact,
        "partial": partial,
        "unmatched_left": unmatched_left,
        "unmatched_right": unmatched_right,
    }


def parse_mentions(raw: str | None) -> list[dict]:
    return json.loads(raw) if raw else []


def agreement_stats(results: pd.DataFrame, arm: str, baseline: str = "full") -> dict:
    """Aggregate alignment statistics of `arm` against `baseline`, per document."""
    left = results[results["arm"] == arm].set_index("reference")
    right = results[results["arm"] == baseline].set_index("reference")
    common = left.index.intersection(right.index)

    n_left = n_right = n_exact = n_partial = 0
    identical = 0
    for reference in common:
        left_mentions = parse_mentions(left.loc[reference, "mentions_json"])
        right_mentions = parse_mentions(right.loc[reference, "mentions_json"])
        outcome = match_mentions(left_mentions, right_mentions)
        n_left += len(left_mentions)
        n_right += len(right_mentions)
        n_exact += len(outcome["exact"])
        n_partial += len(outcome["partial"])
        identical += int(len(outcome["exact"]) == len(left_mentions) == len(right_mentions))

    matched = n_exact + n_partial
    return {
        "documents": len(common),
        "mentions (arm)": n_left,
        "mentions (full)": n_right,
        "exact match (%)": 100 * n_exact / n_left if n_left else float("nan"),
        "exact+partial match (%)": 100 * matched / n_left if n_left else float("nan"),
        "full mentions covered (%)": 100 * matched / n_right if n_right else float("nan"),
        "identical documents (%)": 100 * identical / len(common) if len(common) else float("nan"),
    }


def disagreement_rows(results: pd.DataFrame, cap: int) -> list[dict]:
    """Every mention that could not be aligned with the `full` arm, for manual review."""
    rows = []
    by_arm = {arm: frame.set_index("reference") for arm, frame in results.groupby("arm")}
    if "full" not in by_arm:
        return rows

    references = sorted(by_arm["full"].index)
    for reference in references:
        full_mentions = parse_mentions(by_arm["full"].loc[reference, "mentions_json"])
        for arm, frame in by_arm.items():
            if arm == "full" or reference not in frame.index:
                continue
            mentions = parse_mentions(frame.loc[reference, "mentions_json"])
            outcome = match_mentions(mentions, full_mentions)
            if outcome["unmatched_left"] or outcome["unmatched_right"] or outcome["partial"]:
                rows.append(
                    {
                        "reference": reference,
                        "arm": arm,
                        "n_mentions": len(mentions),
                        "n_full": len(full_mentions),
                        "n_partial": len(outcome["partial"]),
                        "n_unmatched": len(outcome["unmatched_left"]),
                        "n_missing": len(outcome["unmatched_right"]),
                        "mentions": [mentions[i] for i in outcome["unmatched_left"]],
                        "partial_pairs": [
                            (mentions[i], full_mentions[j]) for i, j in outcome["partial"]
                        ],
                        "missing_mentions": [full_mentions[i] for i in outcome["unmatched_right"]],
                    }
                )
    return rows


def format_mention(mention: dict) -> str:
    """Compact one-line rendering of a mention for manual review."""
    paliers = []
    for palier in mention["paliers"]:
        rate = palier.get("taux_valeur")
        unit = (palier.get("type_majoration") or "")[:3]
        paliers.append(
            f"{rate}{unit} [rang {palier.get('rang_debut')}-{palier.get('rang_fin')}"
            f" | vol {palier.get('volume_heures_debut')}-{palier.get('volume_heures_fin')}]"
        )
    citation = re.sub(r"\s+", " ", mention.get("citation_exacte") or "")[:180]
    population = mention.get("population_concernee") or "-"
    return (
        f"- `{mention.get('nature')}` | seuil {mention.get('seuil_declenchement')} | "
        f"{population} | paliers: {'; '.join(paliers) or '-'}\n"
        f"  > {citation}"
    )


def write_review(cap: int, results: pd.DataFrame) -> int:
    """Manual-adjudication file: ambiguous documents with every arm side by side."""
    rows = disagreement_rows(results, cap)
    by_reference: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_reference[row["reference"]].append(row)

    ranked = sorted(
        by_reference.items(),
        key=lambda item: sum(r["n_unmatched"] + r["n_missing"] for r in item[1]),
        reverse=True,
    )

    lines = [
        f"# Manual review — ceiling {cap} chars",
        "",
        "Mentions the matcher could not align with the `full` arm, or aligned only "
        "partially (same nature/threshold/population, different paliers).",
        f"{len(ranked)} document(s) need adjudication; the {min(len(ranked), MAX_REVIEW_DOCS)} "
        "largest disagreements are listed below.",
        "",
    ]
    for reference, arm_rows in ranked[:MAX_REVIEW_DOCS]:
        total = sum(r["n_unmatched"] + r["n_missing"] for r in arm_rows)
        lines += [f"## {reference} — {total} disagreeing mention(s)", ""]
        for row in arm_rows:
            lines += [f"### {row['arm']} ({row['n_mentions']} mentions)", ""]
            if row["partial_pairs"]:
                lines += ["Same nature/threshold/population, different paliers:", ""]
                for arm_mention, full_mention in row["partial_pairs"]:
                    lines.append(f"{row['arm']}: {format_mention(arm_mention).lstrip('- ')}")
                    lines.append(f"full: {format_mention(full_mention).lstrip('- ')}")
                    lines.append("")
            if row["mentions"]:
                lines += ["Only in this arm:", ""]
                lines += [format_mention(mention) for mention in row["mentions"]] + [""]
            if row["missing_mentions"]:
                lines += ["Only in `full` (missing here):", ""]
                lines += [format_mention(mention) for mention in row["missing_mentions"]] + [""]
        lines.append("")

    review_path(cap).write_text("\n".join(lines), encoding="utf-8")
    return len(ranked)


def load_results(cap: int) -> pd.DataFrame:
    frame = pd.read_json(results_path(cap), lines=True)
    return frame[frame["erreur"].isna()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Align the mentions extracted by each arm")
    parser.add_argument(
        "--caps",
        type=str,
        default=",".join(str(cap) for cap in CAPS),
        help="Comma-separated ceilings to review.",
    )
    args = parser.parse_args()

    for cap in (int(value) for value in args.caps.split(",") if value.strip()):
        results = load_results(cap)
        if results.empty:
            print(f"[{cap}] no successful result, skipping")
            continue
        n_documents = write_review(cap, results)
        print(f"[{cap}] {n_documents} document(s) to adjudicate → {review_path(cap)}")


if __name__ == "__main__":
    main()
