"""
Verify every extracted citation against the source document itself.

`run.py` already reports whether a citation is a substring of the context that was
*sent*. That check cannot tell a paraphrase from a fabrication, because the context
text is not stored. This script re-reads the document from the v2 shards (S3 only, no
LLM, no embeddings) and classifies each citation as:

  - `exact`      : present verbatim in the document (whitespace/case insensitive);
  - `partial`    : not verbatim as a whole, but contains a long verbatim run (>= 60
                   characters, about ten words) — the model quoted and stitched;
  - `fabricated` : no verbatim run of that length — the passage, and usually the rule,
                   was invented.

An `exact` citation is necessary but not sufficient: on the flat arms the passage may
exist in the document and not in the context that was sent. Comparing the two verdicts
(`n_citations_grounded` from `run.py` and this file) separates "the model paraphrased"
from "the context never contained the passage".

Each arm is checked against the text it was actually built from: the v2 parts for
`full`, `structured` and `flat_v2`, and the legacy chunks for `flat_legacy` (the two
collections come from different conversions of the same agreement, so the wording is not
identical).

Write access: none. Reads `s3://mateomorin/legifrance/rag/v2/shards` and the Qdrant
collections, read-only.

Usage:
    uv run extraction_benchmark/verify.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
from difflib import SequenceMatcher

import pandas as pd
import s3fs
from _common import CAPS, LEGACY_COLLECTION, citations_path, results_path
from context import render_document
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv(override=True)

SHARDS = "s3://mateomorin/legifrance/rag/v2/shards"
JSON_COLUMNS = [
    "cap", "reference", "arm", "index", "nature", "citation_chars", "verdict", "covered",
]
# A verbatim run of this many characters is long enough not to be a coincidence.
VERBATIM_CHARS = 60


def filesystem() -> s3fs.S3FileSystem:
    """S3 client using the credentials from `.env`."""
    return s3fs.S3FileSystem(
        endpoint_url="https://minio.lab.sspcloud.fr",
        client_kwargs={"region_name": "us-east-1"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=os.environ.get("AWS_SESSION_TOKEN"),
    )


def squeeze(text: str) -> str:
    """Whitespace-insensitive, case-insensitive form used for substring checks."""
    return re.sub(r"\s+", " ", text or "").replace("\u2019", "'").strip().lower()


def document_texts(fs: s3fs.S3FileSystem, references: set[str]) -> dict[str, str]:
    """Normalised title+body text of every requested document, one S3 pass."""
    texts = {}
    root = SHARDS.removeprefix("s3://")
    shards = sorted(path for path in fs.ls(root) if path.endswith(".parquet"))
    for position, shard in enumerate(shards, start=1):
        frame = pd.read_parquet(
            f"s3://{shard}", filesystem=fs, columns=["reference", "parts_json"]
        )
        for reference, raw in zip(frame["reference"], frame["parts_json"]):
            if reference in references and reference not in texts:
                # Same renderer as the pipeline: body + markdown tables + headers. Table
                # rows are stored apart from `body_md`, so a body-only concatenation
                # would make every table-derived citation look fabricated.
                texts[reference] = squeeze(render_document(json.loads(raw), set(), ""))
        print(f"[{position}/{len(shards)}] {len(texts)}/{len(references)} documents")
        if len(texts) == len(references):
            break
    return texts


def legacy_texts(references: set[str]) -> dict[str, str]:
    """Text of the legacy chunks of each document, i.e. what `flat_legacy` could see."""
    client = QdrantClient(
        url=os.environ.get("QDRANT_API_URL", "http://qdrant:6333"),
        api_key=os.environ.get("QDRANT_API_KEY"),
        timeout=120,
    )
    texts = {}
    for position, reference in enumerate(sorted(references), start=1):
        points, _ = client.scroll(
            LEGACY_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.reference", match=models.MatchValue(value=reference)
                    )
                ]
            ),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        chunks = [point.payload.get("page_content") or "" for point in points]
        texts[reference] = squeeze("\n".join(chunks))
        if position % 100 == 0:
            print(f"[legacy] {position}/{len(references)} documents")
    client.close()
    return texts


def best_window(citation: str, text: str) -> str | None:
    """
    Most plausible location of the citation in the document.

    One anchor per third of the citation, the rarest one in the document, and a window of
    the citation's size around each anchor: the best-scoring window wins. Anchoring on the
    rarest word avoids landing on the first common word of a long agreement.
    """
    if not citation or not text:
        return None
    words = re.findall(r"[a-z0-9]{6,}", citation)
    if not words:
        return None
    size = len(words)
    anchors = []
    for start, end in ((0, size // 3), (size // 3, 2 * size // 3), (2 * size // 3, size)):
        chunk = words[start:end] or words[-1:]
        anchors.append(min(chunk, key=text.count))

    best, best_ratio = None, 0.0
    for anchor in anchors:
        position = text.find(anchor)
        if position < 0:
            continue
        window = text[max(0, position - 40) : position + len(citation) + 80]
        similarity = SequenceMatcher(None, citation, window).ratio()
        if similarity > best_ratio:
            best, best_ratio = window, similarity
    return best


def verify_citation(citation: str, text: str) -> tuple[str, float]:
    """Verdict + share of the citation reproduced verbatim, for one citation."""
    normalised = squeeze(citation)
    if not normalised:
        return "empty", 0.0
    if normalised in text:
        return "exact", 1.0

    # The model sometimes joins two passages with an ellipsis: drop the marker, so the
    # verdict measures the quoted passages and not the gap between them.
    joined = squeeze(re.sub(r"\[\.\.\.\]|\.\.\.|…", " ", citation))
    window = best_window(joined, text)
    if window is None:
        return "fabricated", 0.0
    block = SequenceMatcher(None, joined, window).find_longest_match(
        0, len(joined), 0, len(window)
    )
    verdict = "partial" if block.size >= VERBATIM_CHARS else "fabricated"
    return verdict, round(block.size / len(joined), 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify citations against source documents")
    parser.add_argument(
        "--caps",
        type=str,
        default=",".join(str(cap) for cap in CAPS),
        help="Comma-separated ceilings to verify.",
    )
    args = parser.parse_args()

    caps = [int(value) for value in args.caps.split(",") if value.strip()]
    frames = {cap: pd.read_json(results_path(cap), lines=True) for cap in caps}
    references = {ref for frame in frames.values() for ref in frame["reference"]}
    texts = document_texts(filesystem(), references)

    missing = references - set(texts)
    if missing:
        print(f"{len(missing)} reference(s) absent from {SHARDS}: their citations are skipped")

    legacy = legacy_texts(references) if "flat_legacy" in set(
        pd.concat(frames.values())["arm"]
    ) else {}

    for cap, frame in frames.items():
        rows = []
        for row in frame[frame["erreur"].isna()].itertuples():
            source = legacy if row.arm == "flat_legacy" else texts
            text = source.get(row.reference)
            if text is None:
                continue
            mentions = json.loads(row.mentions_json) if row.mentions_json else []
            for index, mention in enumerate(mentions):
                citation = mention.get("citation_exacte") or ""
                verdict, covered = verify_citation(citation, text)
                rows.append(
                    {
                        "cap": cap,
                        "reference": row.reference,
                        "arm": row.arm,
                        "index": index,
                        "nature": mention.get("nature"),
                        "citation_chars": len(citation),
                        "verdict": verdict,
                        "covered": covered,
                    }
                )
        out = pd.DataFrame(rows, columns=JSON_COLUMNS)
        out.to_json(citations_path(cap), orient="records", lines=True, force_ascii=False)
        counts = out["verdict"].value_counts().to_dict() if len(out) else {}
        print(f"[{cap}] {len(out)} citations verified {counts} → {citations_path(cap)}")


if __name__ == "__main__":
    main()
