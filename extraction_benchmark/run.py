"""
Run the four context strategies on the three benchmark samples.

Arms (same prompt, same schema, same model, same temperature for all of them):
  - full         : the whole document, rendered in structural order;
  - structured   : the production hybrid pipeline, forced (anchors + outline selector +
                   similarity + sibling completion), bounded by the ceiling under test;
  - flat_legacy  : chunk similarity on the previous collection, bounded by the ceiling;
  - flat_v2      : chunk similarity on the new parts collection, bounded by the ceiling.

Documents are sampled under 20k, 30k and 40k characters; every arm of a given ceiling
sees exactly the same documents.

Usage:
    uv run extraction_benchmark/run.py --concurrency 5
    uv run extraction_benchmark/run.py --caps 20000 --arms full,structured --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time

import mlflow
import pandas as pd
from _common import (
    ARMS,
    CAPS,
    COLUMNS,
    LEGACY_COLLECTION,
    PROMPTS_DIR,
    V2_COLLECTION,
    build_services,
    expand_subtrees,
    find_anchor_ids,
    load_config,
    load_parts_index,
    preamble_ids,
    render_document,
    results_path,
    samples_path,
)
from dotenv import load_dotenv
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_qdrant import QdrantVectorStore
from mlflow.langchain.langchain_tracer import MlflowLangchainTracer
from qdrant_client.http import models
from schema import ExtractionMajorationsBench
from tqdm.asyncio import tqdm_asyncio

load_dotenv(override=True)
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow_handler = MlflowLangchainTracer()

REF_KEY = "metadata.reference"


class UsageMeter(AsyncCallbackHandler):
    """Sum the token usage of the outline selector, which is a separate LLM call."""

    def __init__(self) -> None:
        """Start from an empty meter."""
        self.input_tokens = 0
        self.output_tokens = 0

    async def on_llm_end(self, response, **kwargs) -> None:
        """Add the usage reported by one LLM call."""
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if usage:
                    self.input_tokens += usage.get("input_tokens", 0)
                    self.output_tokens += usage.get("output_tokens", 0)


async def with_retries(call, attempts: int = 3):
    """Retry transient provider failures (timeouts, rate limits) with a linear backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except Exception:
            if attempt == attempts:
                raise
            await asyncio.sleep(2 * attempt)


def normalize(text: str | None) -> str:
    """Whitespace-insensitive form used for the verbatim-citation check."""
    return re.sub(r"\s+", " ", text or "").strip().lower()


def score_output(parsed, context: str) -> dict:
    """Objective signals: mentions, rates and grounding of the verbatim citations."""
    mentions = parsed.mentions if parsed else []
    norm_context = normalize(context)

    citations = [mention.citation_exacte for mention in mentions if mention.citation_exacte]
    paliers = [palier for mention in mentions for palier in mention.paliers]
    return {
        "n_mentions": len(mentions),
        "n_regles": sum(1 for m in mentions if m.nature == "regle_conventionnelle"),
        "n_rappels": sum(1 for m in mentions if m.nature == "rappel"),
        "n_indeterminees": sum(1 for m in mentions if m.nature == "indeterminee"),
        "n_with_rate": sum(1 for palier in paliers if palier.taux_valeur is not None),
        "n_paliers": len(paliers),
        "n_citations": len(citations),
        "n_citations_grounded": sum(1 for c in citations if normalize(c) in norm_context),
        "mentions_json": json.dumps(
            [mention.model_dump(mode="json") for mention in mentions], ensure_ascii=False
        ),
    }


async def flat_context(
    vector_store: QdrantVectorStore,
    reference: str,
    queries: list[str],
    top_k: int,
    max_chars: int,
) -> tuple[str, int]:
    """
    Previous chunk retrieval, bounded by the ceiling under test.

    Same selection as production: `top_k` chunks for the first query, then one new chunk
    per additional query. Chunks are kept in retrieval order (most relevant first) until
    the budget is reached; a chunk is never truncated.
    """
    from main import retriever_factory

    retrieved = []
    seen: set[str] = set()
    used = 0

    for index, query in enumerate(queries):
        retriever = retriever_factory(
            vector_store=vector_store,
            ref_key=REF_KEY,
            ref_value=reference,
            top_k=top_k if index == 0 else 1,
            exclude_point_ids=list(seen) if seen else None,
        )
        for chunk in await retriever.ainvoke(query):
            chunk_id = chunk.metadata.get("chunk_id")
            if not chunk_id or chunk_id in seen:
                continue
            size = len(chunk.page_content) + 2
            if used + size > max_chars:
                return "\n\n".join(item.page_content for item in retrieved), len(retrieved)
            seen.add(chunk_id)
            retrieved.append(chunk)
            used += size

    return "\n\n".join(item.page_content for item in retrieved), len(retrieved)


TOLERANCE_CHARS = 100


def full_context(parts: list[dict], profile, cap: int) -> tuple[str, int]:
    """
    Whole document, rendered exactly as the production pipeline renders it.

    The sample measures the rendered length without the anchor tag, which the renderer
    then appends to each application-scope heading; a document may therefore exceed the
    ceiling by a few characters. `TOLERANCE_CHARS` absorbs that measurement gap.
    """
    anchor_ids = expand_subtrees(parts, find_anchor_ids(parts, list(profile.anchor_titles)))
    tag = profile.anchor_tag
    if not anchor_ids:
        anchor_ids = expand_subtrees(parts, preamble_ids(parts))
        tag = profile.get("preamble_tag", tag)
    text = render_document(parts, set(anchor_ids), tag)
    if len(text) > cap + TOLERANCE_CHARS:
        raise ValueError(f"Document above the {cap} ceiling: {len(text)} chars")
    return text, len(parts)


async def build_arm_context(
    arm: str,
    reference: str,
    parts: list[dict],
    profile,
    cap: int,
    selector_chain,
    services,
    stores: dict[str, QdrantVectorStore],
    queries: list[str],
    top_k: int,
) -> tuple[str, int, str]:
    """Return (context text, number of context units, mode label)."""
    if arm == "full":
        text, n_context = full_context(parts, profile, cap)
        return text, n_context, "full"

    if arm == "structured":
        from main import structured_context

        text, info = await structured_context(
            ref_value=reference,
            parts=parts,
            retrieval=profile,
            selector_chain=selector_chain,
            queries=queries,
            vector_store=services.vector_store,
            ref_key=REF_KEY,
            force_hybrid=True,
        )
        return text, info["n_context_parts"], info["mode"]

    if arm in ("flat_legacy", "flat_v2"):
        text, n_context = await flat_context(
            stores[arm], reference, queries, top_k=top_k, max_chars=cap
        )
        return text, n_context, "flat"

    raise ValueError(f"Unknown arm: {arm}")


async def run_one(
    sample_row,
    arm: str,
    cap: int,
    chain,
    selector_chain,
    services,
    stores: dict[str, QdrantVectorStore],
    parts_index: dict,
    profile,
    queries: list[str],
    top_k: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    row = {
        "cap": cap,
        "reference": sample_row.reference,
        "anneeSignature": int(sample_row.anneeSignature),
        "arm": arm,
        "mode": None,
        "full_chars": int(sample_row.full_chars),
        "n_parts_doc": int(sample_row.n_parts),
        "context_chars": None,
        "n_context_parts": None,
        "tokens_in": None,
        "tokens_out": None,
        "selector_tokens_in": None,
        "selector_tokens_out": None,
        "context_s": None,
        "llm_s": None,
        "latency_s": None,
        "erreur": None,
    }
    started = time.perf_counter()

    async with semaphore:
        try:
            parts = parts_index[sample_row.reference]
            meter = UsageMeter()
            # The tracer must be attached here too: `with_config(callbacks=...)` scopes
            # callbacks to this runnable, so a selector built with only the meter is
            # invisible in MLflow even though it is a real LLM call.
            selector = (
                selector_chain.with_config(
                    callbacks=[meter, mlflow_handler],
                    run_name=f"outline-selector-{sample_row.reference}",
                    metadata={"reference": sample_row.reference, "arm": arm},
                )
                if selector_chain
                else None
            )

            context_started = time.perf_counter()
            text, n_context, mode = await with_retries(
                lambda: build_arm_context(
                    arm, sample_row.reference, parts, profile, cap, selector, services,
                    stores, queries, top_k,
                )
            )
            row["context_s"] = time.perf_counter() - context_started

            llm_started = time.perf_counter()
            result = await with_retries(
                lambda: chain.ainvoke(
                    {"context": text},
                    config={
                        "callbacks": [mlflow_handler],
                        "run_name": f"extraction-{sample_row.reference}",
                        "metadata": {
                            "reference": sample_row.reference,
                            "n_chunks": n_context,
                            "year": sample_row.anneeSignature,
                            "arm": arm,
                        }
                    }
                )
            )
            row["llm_s"] = time.perf_counter() - llm_started

            raw, parsed = result["raw"], result["parsed"]
            usage = getattr(raw, "usage_metadata", None) or {}
            row.update(
                {
                    "mode": mode,
                    "context_chars": len(text),
                    "n_context_parts": n_context,
                    "tokens_in": usage.get("input_tokens"),
                    "tokens_out": usage.get("output_tokens"),
                    "selector_tokens_in": meter.input_tokens,
                    "selector_tokens_out": meter.output_tokens,
                }
            )
            if parsed is None:
                raise ValueError(f"Structured output parsing failed: {result['parsing_error']}")
            row.update(score_output(parsed, text))
        except Exception as e:
            row["erreur"] = repr(e)[:300]

    row["latency_s"] = time.perf_counter() - started
    return row


def load_done(path) -> set[tuple[str, str]]:
    """
    Successful `(reference, arm)` pairs already in a results file.

    Failed rows are dropped from the file so that `--resume` retries exactly them,
    without producing duplicates.
    """
    if not path.exists():
        return set()
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("erreur") is None:
            kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return {(json.loads(line)["reference"], json.loads(line)["arm"]) for line in kept}


def missing_references(client, collection: str, references: list[str]) -> set[str]:
    """Return the references absent from a collection (they would get an empty context)."""
    missing = set()
    for reference in references:
        count = client.count(
            collection,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=REF_KEY, match=models.MatchValue(value=reference)
                    )
                ]
            ),
            exact=True,
        ).count
        if count == 0:
            missing.add(reference)
    return missing


async def main_async(args) -> None:
    from main import create_prompt, create_queries

    caps = [int(cap) for cap in args.caps.split(",") if cap.strip()]
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"Unknown arm(s): {sorted(unknown)}")

    samples = {cap: pd.read_parquet(samples_path(cap)) for cap in caps}
    if args.limit:
        samples = {cap: frame.head(args.limit) for cap, frame in samples.items()}

    cfg, profile = load_config("majoration", caps[0])
    services = build_services(cfg)

    # Pre-flight: every arm must be able to retrieve from its collection for every
    # sampled reference, otherwise a missing document would be scored as an empty result.
    for cap, frame in samples.items():
        references = frame["reference"].tolist()
        for collection in (LEGACY_COLLECTION, V2_COLLECTION):
            missing = missing_references(services.qdrant_client, collection, references)
            if missing:
                print(f"[{cap}] {len(missing)} ref(s) absent from {collection}: dropped")
                samples[cap] = frame[~frame["reference"].isin(missing)]

    all_references = sorted({ref for frame in samples.values() for ref in frame["reference"]})
    parts_index = load_parts_index(services.fs, profile.parts_path, all_references)

    schema_llm = services.llm.with_structured_output(
        ExtractionMajorationsBench, include_raw=True
    )
    prompt = create_prompt(
        system_prompt_file=str(PROMPTS_DIR / "system_prompt.txt"),
        user_prompt_file=str(PROMPTS_DIR / "user_prompt.txt"),
    )
    chain = prompt | schema_llm

    from _common import SelectedSections

    selector_prompt = create_prompt(
        system_prompt_file=profile.selector_system_prompt_file,
        user_prompt_file=profile.selector_user_prompt_file,
    )
    selector_chain = selector_prompt | services.llm.with_structured_output(SelectedSections)

    stores = {
        "flat_legacy": QdrantVectorStore(
            client=services.qdrant_client,
            collection_name=LEGACY_COLLECTION,
            embedding=services.embeddings,
        ),
        "flat_v2": QdrantVectorStore(
            client=services.qdrant_client,
            collection_name=V2_COLLECTION,
            embedding=services.embeddings,
        ),
    }

    queries = create_queries(cfg.prompts.query_file)
    semaphore = asyncio.Semaphore(args.concurrency)

    for cap in caps:
        sample = samples[cap]
        if sample.empty:
            print(f"[{cap}] no document left after the pre-flight, skipping")
            continue

        done = load_done(results_path(cap)) if args.resume else set()
        if done:
            print(f"[{cap}] {len(done)} run(s) already done, resuming")

        tasks = [
            run_one(
                row, arm, cap, chain, selector_chain, services, stores, parts_index,
                profile, queries, cfg.args.top_k, semaphore,
            )
            for row in sample.itertuples()
            for arm in arms
            if (row.reference, arm) not in done
        ]
        if not tasks:
            print(f"[{cap}] nothing left to run")
            continue
        print(f"[{cap}] {len(sample)} documents x {len(arms)} arms = {len(tasks)} runs")

        path = results_path(cap)
        with path.open("a" if args.resume else "w", encoding="utf-8") as out:
            for future in tqdm_asyncio.as_completed(tasks, desc=f"benchmark {cap}"):
                row = await future
                payload = {column: row.get(column) for column in COLUMNS}
                out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                out.flush()
        print(f"[{cap}] results written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the context strategies for extraction")
    parser.add_argument(
        "--caps",
        type=str,
        default=",".join(str(cap) for cap in CAPS),
        help="Comma-separated document ceilings: 20000, 30000, 40000.",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default=",".join(ARMS),
        help="Comma-separated arms: full, structured, flat_legacy, flat_v2.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only the first N sampled documents.")
    parser.add_argument("--concurrency", type=int, default=5, help="Max simultaneous runs.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep successful runs already in the results file and retry the others.",
    )
    mlflow.set_experiment("benchmark-cap-majoration")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
