import asyncio
import logging
import os
import sys
from pathlib import Path

import hydra
import mlflow
import pandas as pd
from context import (
    SelectedSections,
    build_context,
    build_outline,
    expand_subtrees,
    find_anchor_ids,
    preamble_ids,
    render_document,
)
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from mlflow.langchain.langchain_tracer import MlflowLangchainTracer
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from registry import get_structured_output
from s3fs import S3FileSystem
from services import ServiceContainer
from tqdm.asyncio import tqdm_asyncio

from rag import load_parts_index, reference_filter, similarity_section_ids

SCRIPT_DIR = Path(__file__).parent

load_dotenv(override=True)
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow_handler = MlflowLangchainTracer()

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
for noisy in ("httpx", "httpx2", "httpcore", "qdrant_client", "openai", "s3fs", "fsspec"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------------------------------------------------------- #
# Prompts / query / retrieval profile
# --------------------------------------------------------------------------- #
def create_prompt(system_prompt_file: str, user_prompt_file: str) -> ChatPromptTemplate:
    """
    Retrieve prompts from text files and create a prompt template.
    """
    system_prompt = (SCRIPT_DIR / system_prompt_file).read_text(encoding="utf-8")
    user_prompt = (SCRIPT_DIR / user_prompt_file).read_text(encoding="utf-8")

    return ChatPromptTemplate.from_messages([("system", system_prompt), ("human", user_prompt)])


def create_queries(query_file) -> list[str]:
    """
    Retrieve queries from a text file for similarity search (one query per line).
    """
    lines = (SCRIPT_DIR / query_file).read_text(encoding="utf-8").split("\n")
    return [line.strip() for line in lines if line.strip()]


def load_retrieval(cfg: DictConfig) -> DictConfig:
    """
    Load the retrieval profile of the requested extraction type.

    Profiles live in `configs/retrieval/<type>.yaml`: mode, collection, anchors,
    thresholds and selector prompts. This is what keeps the pipeline itself agnostic
    about the information being extracted.
    """
    path = SCRIPT_DIR / cfg.retrieval_dir / f"{cfg.args.type}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No retrieval profile for type '{cfg.args.type}': {path}")
    return OmegaConf.load(path)


# --------------------------------------------------------------------------- #
# Qdrant : filters, reference listing, and RAG
# --------------------------------------------------------------------------- #
def year_filter(year_key: str, year_value: int) -> models.Filter:
    """
    Basic filter on year key.
    """
    return models.Filter(
        should=[
            models.FieldCondition(key=year_key, match=models.MatchValue(value=str(year_value))),
        ]
    )


async def list_references(
    qdrant_client: QdrantClient,
    qdrant_collection: str,
    ref_key: str,
    year_key: str,
    year_value: int,
    limit: int = 10000
) -> list:
    """
    List all distinct references after QDrant Facet.
    Make sure to have reference as an index of your Qdrant Database.
    """
    flt = year_filter(year_key=year_key, year_value=year_value)
    response = await asyncio.to_thread(
            qdrant_client.facet,
            key=ref_key,
            collection_name=qdrant_collection,
            facet_filter=flt,
            limit=limit,
        )

    hits = response.hits

    refs = [hit.value for hit in hits]
    n_points = sum([hit.count for hit in hits])

    logger.info(f"[{year_value}] {n_points} chunks scanned → {len(refs)} distinct references")

    return sorted(refs, key=str)


def retriever_factory(
    vector_store: VectorStore,
    ref_key: str,
    ref_value: str,
    top_k: int,
    exclude_point_ids: list[str | int] | None = None
) -> VectorStoreRetriever:
    """
    RAG for the specific reference
    """
    flt = reference_filter(ref_key=ref_key, ref_value=ref_value)

    if exclude_point_ids:
        flt.must_not = [
            models.HasIdCondition(has_id=exclude_point_ids)
        ]

    return vector_store.as_retriever(
        search_kwargs={"k": top_k, "filter": flt}
    )


# --------------------------------------------------------------------------- #
# Structured context (anchors + outline selector + similarity)
# --------------------------------------------------------------------------- #
async def structured_context(
    ref_value: str,
    parts: list[dict],
    retrieval: DictConfig,
    selector_chain: Runnable | None,
    queries: list[str],
    vector_store: VectorStore,
    ref_key: str,
    force_hybrid: bool = False,
) -> tuple[str, dict]:
    """
    Context of one document: whole document when it fits, hybrid retrieval otherwise.

    Full reading is bounded by `max_document_chars`; the hybrid context is bounded by
    `max_context_chars` so that it never exceeds what a full reading would have cost.
    `force_hybrid` skips the full-reading shortcut (used to benchmark the hybrid path
    on documents that would otherwise be read in full).
    """
    anchor_ids = expand_subtrees(
        parts, find_anchor_ids(parts, list(retrieval.anchor_titles))
    )
    anchor_source = "title"
    anchor_tag = retrieval.anchor_tag
    if not anchor_ids:
        anchor_ids = expand_subtrees(parts, preamble_ids(parts))
        anchor_source = "preamble" if anchor_ids else "none"
        anchor_tag = retrieval.get("preamble_tag", anchor_tag)
    anchor_set = set(anchor_ids)

    full_text = render_document(parts, anchor_set, anchor_tag)
    if not force_hybrid and len(full_text) <= retrieval.max_document_chars:
        return full_text, {
            "mode": "full",
            "anchor_source": anchor_source,
            "n_parts": len(parts),
            "n_anchor_parts": len(anchor_ids),
            "n_selected_parts": 0,
            "n_sibling_parts": 0,
            "n_similarity_parts": 0,
            "n_context_parts": len(parts),
            "n_outline_chars": 0,
            "context_chars": len(full_text),
        }

    if selector_chain is None:
        raise ValueError("The 'structured' retrieval mode requires a selector prompt.")

    outline = build_outline(parts, retrieval.max_outline_chars)
    selection = await selector_chain.ainvoke({"outline": outline})
    known = {part["section_id"] for part in parts}
    selected_ids = [sid for sid in (selection.section_ids if selection else []) if sid in known]

    similarity_ids = await similarity_section_ids(
        vector_store=vector_store,
        ref_key=ref_key,
        ref_value=ref_value,
        queries=queries,
        top_k=retrieval.similarity_top_k,
    )
    similarity_ids = similarity_ids[: retrieval.max_similarity_parts]

    text, info = build_context(
        parts,
        anchor_ids=anchor_ids,
        selected_ids=selected_ids,
        similarity_ids=similarity_ids,
        max_chars=retrieval.max_context_chars,
        max_siblings=retrieval.max_siblings,
        anchor_tag=anchor_tag,
    )
    info["mode"] = "hybrid"
    info["anchor_source"] = anchor_source
    info["n_parts"] = len(parts)
    info["n_outline_chars"] = len(outline)
    return text, info


async def flat_context(
    ref_value: str,
    queries: list[str],
    vector_store: VectorStore,
    top_k: int,
    ref_key: str,
) -> tuple[str, dict]:
    """
    Retrieve chunks based on similarity only.

    Mutliple queries allowed, first with top_k retrieved chunks, the next ones with one chunk.
    """
    retrieved_chunks = []
    seen_point_ids = set()
    info = dict()

    for i, query in enumerate(queries):
        current_k = top_k if i == 0 else 1
        retriever = retriever_factory(
            vector_store=vector_store,
            ref_key=ref_key,
            ref_value=ref_value,
            top_k=current_k,
            exclude_point_ids=list(seen_point_ids) if seen_point_ids else None
        )

        chunks = await retriever.ainvoke(query)
        for chunk in chunks:
            point_id = chunk.metadata["chunk_id"]

            if point_id not in seen_point_ids:
                seen_point_ids.add(point_id)
                retrieved_chunks.append(chunk)

    info["n_chunks"] = len(retrieved_chunks)
    info["context_chars"] = sum(len(chunk.page_content) for chunk in retrieved_chunks)

    return chunk, info


# --------------------------------------------------------------------------- #
# Extraction for one reference
# --------------------------------------------------------------------------- #
async def extract_one(
    ref_key: str,
    ref_value: str,
    year: int,
    chain: Runnable,
    selector_chain: Runnable | None,
    queries: list[str],
    top_k: int,
    vector_store: VectorStore,
    structured_output: BaseModel,
    semaphore: asyncio.Semaphore,
    retrieval: DictConfig,
    parts_index: dict[str, list[dict]],
) -> dict:
    """
    Extract data from one document reference.

    - `structured` mode: structural context (anchors, selector, similarity).
    - `similarity` mode: top_k chunks of the first query, then one unique chunk per
      subsequent query (legacy behaviour, used by the other extraction types).
    """
    extracted_data = {field_name: None for field_name in structured_output.model_fields}
    row = {
        "reference": ref_value,
        "anneeSignature": year,
        "erreur": None,
        **extracted_data
    }
    async with semaphore:
        try:
            if retrieval.mode == "structured":
                parts = parts_index.get(ref_value)
                if not parts:
                    row["erreur"] = "Aucune section retrouvée pour cette référence."
                    return row

                context_text, info = await structured_context(
                    ref_value=ref_value,
                    parts=parts,
                    retrieval=retrieval,
                    selector_chain=selector_chain,
                    queries=queries,
                    vector_store=vector_store,
                    ref_key=ref_key,
                )
                row.update(info)
                n_context = info["n_context_parts"]

            else:
                row["n_chunks"] = 0
                if not queries:
                    row["erreur"] = "Aucune requête fournie."
                    return row

                retrieved_chunks, info = await flat_context(
                    ref_value=ref_value,
                    queries=queries,
                    vector_store=vector_store,
                    top_k=top_k,
                    ref_key=ref_key,
                )
                row.update(info)

                if not retrieved_chunks:
                    row["erreur"] = "Aucun chunk retrouvé pour cette référence."
                    return row

                context_text = "\n\n".join(chunk.page_content for chunk in retrieved_chunks)
                n_context = len(retrieved_chunks)

            res = await chain.ainvoke(
                {"context": context_text},
                config={
                    "callbacks": [mlflow_handler],
                    "run_name": f"extraction-{ref_value}",
                    "metadata": {
                        "reference": ref_value,
                        "n_chunks": n_context,
                        "year": year
                    }
                }
            )
            if res is None:
                raise ValueError("Sortie structurée vide (parsing LLM échoué)")

            res_dict = res.model_dump(mode="json")
            row.update(res_dict)

        except Exception as e:
            logger.error(f"[{year}] Error with {ref_value}: {e!r}")
            row["erreur"] = repr(e)[:500]

    return row


# --------------------------------------------------------------------------- #
# Extraction for one year - utils
# --------------------------------------------------------------------------- #
def remove_treated_data(fs: S3FileSystem, ref_batch: list, shard_path: str):
    """
    Search if already-treated data exist and remove it from the batch to process.
    """
    treated_data = pd.DataFrame()
    if fs.exists(shard_path):
        previous_data = pd.read_parquet(shard_path, filesystem=fs)
        treated_data = previous_data[previous_data["erreur"].isna()]
        ref_batch = [ref for ref in ref_batch if ref not in set(treated_data["reference"])]

    return ref_batch, treated_data


async def process_shard(
    shard: int,
    ref_key: str,
    ref_batch: list,
    year: int,
    extraction_dir: str,
    n_shards: int,
    chain: Runnable,
    selector_chain: Runnable | None,
    services: ServiceContainer,
    queries: list[str],
    top_k: int,
    structured_output: BaseModel,
    semaphore: asyncio.Semaphore,
    retrieval: DictConfig,
    parts_index: dict[str, list[dict]],
) -> pd.DataFrame:
    """
    Treat a single shard with possible cache/restart.
    """
    # Output for the shard
    shard_path = f"{extraction_dir}/shards/shard_{shard:05d}.parquet"

    # Check for already-treated data
    ref_batch, treated_data = remove_treated_data(
        fs=services.fs,
        ref_batch=ref_batch,
        shard_path=shard_path
    )
    if not ref_batch:
        logger.info(f"[{year}] shard {shard + 1}/{n_shards} already full.")

        return treated_data

    # Process
    tasks = [
        extract_one(
            ref_key=ref_key,
            ref_value=ref,
            year=year,
            chain=chain,
            selector_chain=selector_chain,
            queries=queries,
            top_k=top_k,
            structured_output=structured_output,
            vector_store=services.vector_store,
            semaphore=semaphore,
            retrieval=retrieval,
            parts_index=parts_index,
        )
        for ref in ref_batch
    ]
    new_rows = await tqdm_asyncio.gather(
        *tasks, desc=f"[{year}] shard {shard + 1}/{n_shards}"
    )

    # Export
    shard_df = pd.concat([treated_data, pd.DataFrame(new_rows)], ignore_index=True)
    shard_df.to_parquet(path=shard_path, filesystem=services.fs)

    return shard_df


def finalize_year(full_data: pd.DataFrame, year: int, final_path: str, fs: S3FileSystem) -> int:
    """
    Export the final data and show characteristics
    """
    n_err = int(full_data["erreur"].notna().sum())

    logger.info(f"[{year}] {len(full_data)} references | {n_err} errors")

    if n_err == 0:
        full_data.to_parquet(path=final_path, filesystem=fs)
        logger.info(f"[{year}] Final result has been exported to: {final_path}")
    else:
        logger.warning(f"[{year}] Export has been canceled ({n_err} errors).")

    return n_err


# --------------------------------------------------------------------------- #
# Extraction for one year - main script
# --------------------------------------------------------------------------- #
async def process_year(
    cfg: DictConfig,
    services: ServiceContainer,
    chain: Runnable,
    selector_chain: Runnable | None,
    queries: list[str],
    structured_output: BaseModel,
    retrieval: DictConfig,
) -> int:
    """
    Returns the number of references that failed.
    """
    args = cfg.args
    extraction_dir = cfg.data_storage.extraction_dir.strip('/')

    # Final path creation (if not exists)
    final_path = f"{extraction_dir}/extraction_{args.year}.parquet"
    if args.skip_existing and services.fs.exists(final_path):
        logger.info(f"[{args.year}] {final_path} already exists, skip.")
        return 0

    # References listing
    refs = await list_references(
        qdrant_client=services.qdrant_client,
        qdrant_collection=cfg.vector_storage.qdrant_collection,
        ref_key=cfg.vector_storage.ref_key,
        year_key=cfg.vector_storage.year_key,
        year_value=args.year
    )
    if args.limit:
        refs = refs[: args.limit]
    if not refs:
        logger.warning(f"[{args.year}] No reference found.")
        return 0

    # Structural parts of the year (structured mode only): one S3 pass, bounded memory
    parts_index: dict[str, list[dict]] = {}
    if retrieval.mode == "structured":
        parts_index = await asyncio.to_thread(
            load_parts_index,
            services.fs,
            retrieval.parts_path,
            refs,
        )

    # Runing shards
    semaphore = asyncio.Semaphore(args.concurrency)
    shard_dfs = []
    n_shards = (len(refs) + args.shard_size - 1) // args.shard_size

    for shard in range(n_shards):
        ref_batch = refs[shard * args.shard_size: (shard + 1) * args.shard_size]
        shard_df = await process_shard(
            shard=shard,
            ref_key=cfg.vector_storage.ref_key,
            ref_batch=ref_batch,
            year=args.year,
            extraction_dir=extraction_dir,
            n_shards=n_shards,
            chain=chain,
            selector_chain=selector_chain,
            services=services,
            queries=queries,
            top_k=args.top_k,
            structured_output=structured_output,
            semaphore=semaphore,
            retrieval=retrieval,
            parts_index=parts_index,
        )
        shard_dfs.append(shard_df)

    full_data = pd.concat(shard_dfs, ignore_index=True)
    n_err = finalize_year(
        full_data=full_data,
        year=args.year,
        final_path=final_path,
        fs=services.fs
    )

    return n_err


# --------------------------------------------------------------------------- #
# Main functions
# --------------------------------------------------------------------------- #
async def main_async(cfg: DictConfig) -> int:
    # Retrieval profile of the requested type (mode, collection, anchors, ...)
    retrieval = load_retrieval(cfg)
    if retrieval.get("collection"):
        cfg.vector_storage.qdrant_collection = retrieval.collection

    # Connection to external services
    services = ServiceContainer(cfg)

    # Set-up LLM with structured output
    structured_output = get_structured_output(cfg.args.type)
    structured_llm = services.llm.with_structured_output(structured_output)

    # Prompt
    prompts = create_prompt(
        system_prompt_file=cfg.prompts.system_prompt_file,
        user_prompt_file=cfg.prompts.user_prompt_file
    )

    # LLM Ready
    chain = (prompts | structured_llm).with_retry(
        stop_after_attempt=cfg.args.llm_retries, wait_exponential_jitter=True
    )

    # Outline selector (structured mode only)
    selector_chain = None
    if retrieval.mode == "structured":
        selector_prompt = create_prompt(
            system_prompt_file=retrieval.selector_system_prompt_file,
            user_prompt_file=retrieval.selector_user_prompt_file,
        )
        selector_llm = services.llm.with_structured_output(SelectedSections)
        selector_chain = (selector_prompt | selector_llm).with_retry(
            stop_after_attempt=cfg.args.llm_retries, wait_exponential_jitter=True
        )

    # Extraction ready
    queries = create_queries(query_file=cfg.prompts.query_file)

    # Exctract data for the desired year
    mlflow.set_experiment(f"extraction-{cfg.args.type}-{cfg.args.year}")
    total_errors = await process_year(
        cfg=cfg,
        services=services,
        chain=chain,
        selector_chain=selector_chain,
        queries=queries,
        structured_output=structured_output,
        retrieval=retrieval,
    )

    return total_errors


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    n_errors = asyncio.run(main_async(cfg))
    if n_errors:
        logger.error(f"{n_errors} references with errors → exit 1 (Argo will start the pod again).")
        sys.exit(1)


if __name__ == "__main__":
    main()
