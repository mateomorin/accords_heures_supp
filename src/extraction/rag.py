"""I/O helpers for structured extraction: parts index and similarity search."""

from __future__ import annotations

import json
import logging

import pandas as pd
from langchain_core.vectorstores import VectorStore
from qdrant_client.http import models

logger = logging.getLogger(__name__)


def load_parts_index(
    fs,
    parts_path: str,
    refs: list[str],
) -> dict[str, list[dict]]:
    """
    Read the v2 shards once and keep the parts of the requested references.

    The full corpus is ~560 MB of text, but only one year is kept in memory
    (~100 MB for 3.8k documents), which bounds the resident footprint.
    """
    wanted = set(refs)
    index: dict[str, list[dict]] = {}
    root = parts_path.removeprefix("s3://").rstrip("/")
    shards = sorted(path for path in fs.ls(root) if path.endswith(".parquet"))

    for position, shard in enumerate(shards, start=1):
        frame = pd.read_parquet(
            f"s3://{shard}", filesystem=fs, columns=["reference", "parts_json"]
        )
        frame = frame[frame["reference"].isin(wanted)]
        for reference, parts_json in zip(frame["reference"], frame["parts_json"]):
            index[reference] = json.loads(parts_json)
        logger.info(
            f"[parts] {position}/{len(shards)} {shard.rsplit('/', 1)[-1]} "
            f"-> {len(index)}/{len(wanted)} references"
        )

    missing = wanted - set(index)
    if missing:
        logger.warning(f"[parts] {len(missing)} reference(s) absent from {parts_path}")
    return index


def reference_filter(ref_key: str, ref_value: str) -> models.Filter:
    """Filter on one agreement reference."""
    return models.Filter(
        must=[models.FieldCondition(key=ref_key, match=models.MatchValue(value=ref_value))]
    )


async def similarity_section_ids(
    vector_store: VectorStore,
    ref_key: str,
    ref_value: str,
    queries: list[str],
    top_k: int,
) -> list[str]:
    """Section ids ranked by their best chunk score, deduplicated per section."""
    filt = reference_filter(ref_key=ref_key, ref_value=ref_value)
    scores: dict[str, float] = {}

    for query in queries:
        hits = await vector_store.asimilarity_search_with_score(query, k=top_k, filter=filt)
        for document, score in hits:
            section_id = document.metadata.get("section_id")
            if not section_id:
                continue
            previous = scores.get(section_id)
            if previous is None or score > previous:
                scores[section_id] = score

    return sorted(scores, key=scores.get, reverse=True)


async def section_scores(
    vector_store: VectorStore,
    ref_key: str,
    ref_value: str,
    query: str,
    limit: int,
) -> dict[str, float]:
    """
    Best cosine score of each section of one document for one query.

    Used to rank sibling candidates: only the `limit` best chunks of the document are
    scored, so a section that was not retrieved simply has no entry (`-1.0` in the
    ranking). `limit` must therefore cover the chunks of the document, not its parts.
    """
    hits = await vector_store.asimilarity_search_with_score(
        query, k=limit, filter=reference_filter(ref_key=ref_key, ref_value=ref_value)
    )
    scores: dict[str, float] = {}
    for document, score in hits:
        section_id = document.metadata.get("section_id")
        if not section_id:
            continue
        previous = scores.get(section_id)
        if previous is None or score > previous:
            scores[section_id] = score
    return scores
