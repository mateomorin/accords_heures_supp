"""
Build the three benchmark samples: documents under 20k, 30k and 40k characters.

For each ceiling, 25 documents per signature year (2018-2025) are taken at evenly spaced
quantiles of the rendered document size *among the documents that fit under the
ceiling*. The three samples are therefore independent: the 20k sample is not a subset
of the 30k one.

The selection is deterministic (no random seed involved).

Usage:
    uv run extraction_benchmark/sample.py
"""

from __future__ import annotations

import json

import pandas as pd
import s3fs
from _common import CAPS, samples_path
from context import render_document

PARTS_PATH = "s3://mateomorin/legifrance/rag/v2/shards"
YEARS = list(range(2018, 2026))
PER_YEAR = 25


def full_chars(parts: list[dict]) -> int:
    return len(render_document(parts, set(), ""))


def read_corpus(fs: s3fs.S3FileSystem) -> pd.DataFrame:
    frames = []
    root = PARTS_PATH.removeprefix("s3://")
    shards = sorted(path for path in fs.ls(root) if path.endswith(".parquet"))
    for position, shard in enumerate(shards, start=1):
        frame = pd.read_parquet(
            f"s3://{shard}",
            filesystem=fs,
            columns=["reference", "anneeSignature", "n_parts", "parts_json"],
        )
        frame = frame[frame["anneeSignature"].isin(YEARS)]
        frame["full_chars"] = frame["parts_json"].map(lambda raw: full_chars(json.loads(raw)))
        frames.append(frame[["reference", "anneeSignature", "n_parts", "full_chars"]])
        print(f"[{position}/{len(shards)}] {shard.rsplit('/', 1)[-1]}: {len(frame)} docs kept")
    return pd.concat(frames, ignore_index=True)


def sample(corpus: pd.DataFrame, cap: int) -> pd.DataFrame:
    """25 documents per year, at evenly spaced quantiles of size under the ceiling."""
    # Documents whose parts render to nothing carry no rule at all: they would only
    # produce empty extractions in every arm.
    eligible = corpus[(corpus["full_chars"] <= cap) & (corpus["full_chars"] > 0)]
    picked = []
    for year in YEARS:
        frame = eligible[eligible["anneeSignature"] == year].sort_values("full_chars")
        if frame.empty:
            raise ValueError(f"No document under {cap} chars for {year}")
        n = min(PER_YEAR, len(frame))
        positions = [round(index * (len(frame) - 1) / (n - 1)) for index in range(n)]
        year_sample = frame.iloc[sorted(set(positions))].copy()
        year_sample["rank_in_year"] = range(len(year_sample))
        picked.append(year_sample)
    return pd.concat(picked, ignore_index=True)


def main() -> None:
    fs = s3fs.S3FileSystem(
        endpoint_url="https://minio.lab.sspcloud.fr",
        client_kwargs={"region_name": "us-east-1"},
    )
    corpus = read_corpus(fs)
    print(f"{len(corpus)} documents from {YEARS[0]} to {YEARS[-1]}")

    for cap in CAPS:
        frame = sample(corpus, cap)
        frame.to_parquet(samples_path(cap), index=False)
        summary = frame.groupby("anneeSignature")["full_chars"].agg(
            ["min", "median", "max", "count"]
        )
        print(f"\n=== max {cap} chars: {len(frame)} documents ===")
        print(summary.to_string())
        print(f"written to {samples_path(cap)}")


if __name__ == "__main__":
    main()
