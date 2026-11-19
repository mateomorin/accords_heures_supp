"""
Order-insensitive comparison of the mentions produced by the four arms.

The LLM may return the same rules in a different order, split or merge mentions, attach
the same rate to a different population, or describe the same rule with another nature.
A single "exact match" rate therefore mixes three very different things: real content
loss, differences in palier boundaries, and pure labelling noise. Mentions are aligned
in three tiers, from the most specific to the loosest:

  1. `exact`     : same nature, threshold, population and palier set;
  2. `same_rule` : same palier set, but another nature / threshold / population wording
                   (the rule is identical, only its labels differ);
  3. `bounds`    : same nature / threshold / population, but a different palier set
                   (boundaries split, merged or shifted).

What cannot be aligned is left as `arm_only` (an extra rule, possibly a hallucination)
or `full_only` (a rule the arm missed). Those two are the only tiers that represent a
real content divergence, and they are written to `manual_review_<cap>.md` for
adjudication by hand.

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
    """
    Order-insensitive identity of one palier.

    Values are stringified so the tuple is totally ordered: `type_majoration` is
    `None` on a reminder without a rate, and sorting `None` against `"pourcentage"`
    raises `TypeError`.
    """
    return tuple(
        str(value)
        for value in (
            palier.get("type_majoration"),
            number(palier.get("taux_valeur")),
            number(palier.get("rang_debut")),
            number(palier.get("rang_fin")),
            number(palier.get("volume_heures_debut")),
            number(palier.get("volume_heures_fin")),
        )
    )


def palier_set(mention: dict) -> tuple:
    """Identity of the paliers of a mention, ignoring everything else."""
    return tuple(sorted(palier_key(palier) for palier in mention["paliers"]))


def mention_key(mention: dict) -> tuple:
    """Identity of a mention without its paliers (labels only)."""
    return (
        mention.get("nature"),
        str(number(mention.get("seuil_declenchement"))),
        normalize(mention.get("population_concernee")),
    )


def mention_signature(mention: dict) -> tuple:
    """Full, order-insensitive identity of a mention."""
    return mention_key(mention) + (palier_set(mention),)


def _align(left: list[dict], right: list[dict], left_indexes, right_indexes, key):
    """Pair left and right mentions that share the same `key`, first come first served."""
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for index in right_indexes:
        buckets[key(right[index])].append(index)

    pairs = []
    leftovers = []
    for index in left_indexes:
        bucket = buckets.get(key(left[index]))
        if bucket:
            pairs.append((index, bucket.pop(0)))
        else:
            leftovers.append(index)
    return pairs, leftovers


def match_mentions(left: list[dict], right: list[dict]) -> dict:
    """Align two mention lists through the three tiers, then report the leftovers."""
    exact, remaining_left = _align(
        left, right, range(len(left)), list(range(len(right))), mention_signature
    )
    matched_right = {j for _, j in exact}
    remaining_right = [j for j in range(len(right)) if j not in matched_right]

    same_rule, remaining_left = _align(
        left, right, remaining_left, remaining_right, palier_set
    )
    matched_right |= {j for _, j in same_rule}
    remaining_right = [j for j in range(len(right)) if j not in matched_right]

    bounds, remaining_left = _align(
        left, right, remaining_left, remaining_right, mention_key
    )
    matched_right |= {j for _, j in bounds}

    return {
        "exact": exact,
        "same_rule": same_rule,
        "bounds": bounds,
        "arm_only": remaining_left,
        "full_only": [j for j in range(len(right)) if j not in matched_right],
    }


def parse_mentions(raw: str | None) -> list[dict]:
    """Mentions stored as JSON in one result row."""
    return json.loads(raw) if isinstance(raw, str) and raw else []


def compare_documents(results: pd.DataFrame, arm: str, baseline: str = "full") -> dict:
    """Per-arm alignment counts against the baseline, over the documents both cover."""
    left = results[results["arm"] == arm].set_index("reference")
    right = results[results["arm"] == baseline].set_index("reference")
    common = left.index.intersection(right.index)

    counts = dict.fromkeys(("exact", "same_rule", "bounds", "arm_only", "full_only"), 0)
    identical_docs = 0
    docs = 0
    per_document = {}
    for reference in common:
        arm_mentions = parse_mentions(left.loc[reference, "mentions_json"])
        base_mentions = parse_mentions(right.loc[reference, "mentions_json"])
        outcome = match_mentions(arm_mentions, base_mentions)
        docs += 1
        for tier in counts:
            counts[tier] += len(outcome[tier])
        identical_docs += int(len(outcome["exact"]) == len(arm_mentions) == len(base_mentions))
        per_document[reference] = {
            "n_arm": len(arm_mentions),
            "n_full": len(base_mentions),
            "content_divergence": len(outcome["bounds"])
            + len(outcome["arm_only"])
            + len(outcome["full_only"]),
        }

    n_arm = sum(item["n_arm"] for item in per_document.values())
    n_full = sum(item["n_full"] for item in per_document.values())
    aligned = counts["exact"] + counts["same_rule"] + counts["bounds"]
    return {
        "documents": docs,
        **counts,
        "mentions (arm)": n_arm,
        "mentions (full)": n_full,
        "identical (%)": 100 * counts["exact"] / n_arm if n_arm else float("nan"),
        "same rule, other labels (%)": 100 * counts["same_rule"] / n_arm if n_arm else float("nan"),
        "different bounds (%)": 100 * counts["bounds"] / n_arm if n_arm else float("nan"),
        "extra rule (%)": 100 * counts["arm_only"] / n_arm if n_arm else float("nan"),
        "full covered (%)": 100 * aligned / n_full if n_full else float("nan"),
        "documents fully identical (%)": 100 * identical_docs / docs if docs else float("nan"),
        "documents with a content divergence": sum(
            item["content_divergence"] > 0 for item in per_document.values()
        ),
        "_per_document": per_document,
    }


def agreement_stats(results: pd.DataFrame, arm: str, baseline: str = "full") -> dict:
    """Public view of `compare_documents`, without the per-document payload."""
    return {key: value for key, value in compare_documents(results, arm, baseline).items()
            if not key.startswith("_")}


def has_rate(mention: dict) -> bool:
    """Return True when the mention carries at least one palier with a numeric rate."""
    return any(palier.get("taux_valeur") is not None for palier in mention["paliers"])


def divergence_rows(results: pd.DataFrame) -> list[dict]:
    """Every mention that could not be aligned exactly, for manual review."""
    rows = []
    by_arm = {arm: frame.set_index("reference") for arm, frame in results.groupby("arm")}
    if "full" not in by_arm:
        return rows

    for reference in sorted(by_arm["full"].index):
        full_mentions = parse_mentions(by_arm["full"].loc[reference, "mentions_json"])
        for arm, frame in by_arm.items():
            if arm == "full" or reference not in frame.index:
                continue
            mentions = parse_mentions(frame.loc[reference, "mentions_json"])
            outcome = match_mentions(mentions, full_mentions)
            if outcome["arm_only"] or outcome["full_only"] or outcome["bounds"]:
                rows.append(
                    {
                        "reference": reference,
                        "arm": arm,
                        "n_mentions": len(mentions),
                        "n_full": len(full_mentions),
                        "bounds_pairs": [
                            (mentions[i], full_mentions[j]) for i, j in outcome["bounds"]
                        ],
                        "label_pairs": [
                            (mentions[i], full_mentions[j]) for i, j in outcome["same_rule"]
                        ],
                        "arm_only": [mentions[i] for i in outcome["arm_only"]],
                        "full_only": [full_mentions[j] for j in outcome["full_only"]],
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
    rows = divergence_rows(results)
    by_reference: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_reference[row["reference"]].append(row)

    def divergence(row: dict) -> int:
        return len(row["bounds_pairs"]) + len(row["arm_only"]) + len(row["full_only"])

    ranked = sorted(
        by_reference.items(),
        key=lambda item: sum(divergence(row) for row in item[1]),
        reverse=True,
    )

    lines = [
        f"# Manual review — ceiling {cap} chars",
        "",
        "Mentions that the matcher could not align exactly with the `full` arm. "
        "`bounds` pairs are the same rule with different palier boundaries; "
        "`arm only` / `full only` are rules present in one arm and absent from the other; "
        "`labels` pairs are the same rule described with another nature, threshold or "
        "population (labelling noise, listed last, usually no adjudication needed).",
        f"{len(ranked)} document(s) diverge; the {min(len(ranked), MAX_REVIEW_DOCS)} "
        "largest divergences are listed below.",
        "",
    ]
    for reference, arm_rows in ranked[:MAX_REVIEW_DOCS]:
        total = sum(divergence(row) for row in arm_rows)
        lines += [f"## {reference} — {total} mention(s) not exactly aligned", ""]
        for row in arm_rows:
            lines += [f"### {row['arm']} ({row['n_mentions']} mentions)", ""]
            if row["bounds_pairs"]:
                lines += ["Different palier boundaries:", ""]
                for arm_mention, full_mention in row["bounds_pairs"]:
                    lines.append(f"{row['arm']}: {format_mention(arm_mention).lstrip('- ')}")
                    lines.append(f"full: {format_mention(full_mention).lstrip('- ')}")
                    lines.append("")
            if row["arm_only"]:
                lines += ["Only in this arm:", ""]
                lines += [format_mention(mention) for mention in row["arm_only"]] + [""]
            if row["full_only"]:
                lines += ["Only in `full` (missing here):", ""]
                lines += [format_mention(mention) for mention in row["full_only"]] + [""]
            if row["label_pairs"]:
                lines += ["Same rule, other labels:", ""]
                for arm_mention, full_mention in row["label_pairs"]:
                    lines.append(
                        f"  `{arm_mention.get('nature')}` / seuil "
                        f"{arm_mention.get('seuil_declenchement')} / "
                        f"{arm_mention.get('population_concernee') or '-'}  vs  "
                        f"`{full_mention.get('nature')}` / seuil "
                        f"{full_mention.get('seuil_declenchement')} / "
                        f"{full_mention.get('population_concernee') or '-'}"
                    )
                lines.append("")
        lines.append("")

    review_path(cap).write_text("\n".join(lines), encoding="utf-8")
    return len(ranked)


def load_results(cap: int) -> pd.DataFrame:
    """Successful rows of one ceiling."""
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
