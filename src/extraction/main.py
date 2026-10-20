import asyncio
import logging
import sys
from pathlib import Path

import hydra
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from omegaconf import DictConfig
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from s3fs import S3FileSystem
from tqdm.asyncio import tqdm_asyncio

from registry import get_structured_output
from services import ServiceContainer


SCRIPT_DIR = Path(__file__).parent

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
for noisy in ("httpx", "httpx2", "httpcore", "qdrant_client", "openai", "s3fs", "fsspec"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------------------------------------------------------- #
# Prompts / query
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
    Retrieve query from text file for similarity search.
    """
    return (SCRIPT_DIR / query_file).read_text(encoding="utf-8").split("\n")


# --------------------------------------------------------------------------- #
# Qdrant : filters, reference listing, and RAG
# --------------------------------------------------------------------------- #
def reference_filter(ref_key: str, ref_value: str) -> models.Filter:
    """
    Basic filter on reference key.
    """

    return models.Filter(
        must=[models.FieldCondition(key=ref_key, match=models.MatchValue(value=ref_value))]
    )


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
# Extraction for one reference
# --------------------------------------------------------------------------- #
async def extract_one(
    ref_key: str,
    ref_value: str,
    year: int,
    chain: Runnable,
    queries: list[str],
    top_k: int,
    vector_store: VectorStore,
    structured_output: BaseModel,
    semaphore: asyncio.Semaphore
) -> dict:
    """
    Extract data from the document reference using multiple queries.
    - First query fetches top_k chunks.
    - Subsequent queries fetch 1 unique chunk each (if not already retrieved).
    """
    extracted_data = {field_name: None for field_name in structured_output.model_fields}
    row = {
        "reference": ref_value,
        "anneeSignature": year,
        "n_chunks": 0,
        "erreur": None,
        **extracted_data
    }
    async with semaphore:
        try:
            if not queries:
                row["erreur"] = "Aucune requête fournie."
                return row

            # Chunk id = point id
            retrieved_chunks = []
            seen_point_ids = set()
            point_id = 0

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

            row["n_chunks"] = len(retrieved_chunks)

            if not retrieved_chunks:
                row["erreur"] = "Aucun chunk retrouvé pour cette référence."
                return row

            context_text = "\n\n".join(d.page_content for d in chunks)
            res = await chain.ainvoke({"context": context_text})
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
    services: ServiceContainer,
    queries: list[str],
    top_k: int,
    structured_output: BaseModel,
    semaphore: asyncio.Semaphore,
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
            queries=queries,
            top_k=top_k,
            structured_output=structured_output,
            vector_store=services.vector_store,
            semaphore=semaphore)
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
    queries: list[str],
    structured_output: BaseModel
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
            services=services,
            queries=queries,
            top_k=args.top_k,
            structured_output=structured_output,
            semaphore=semaphore
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

    # Extraction ready
    queries = create_queries(query_file=cfg.prompts.query_file)

    # Exctract data for the desired year
    total_errors = await process_year(
        cfg=cfg,
        services=services,
        chain=chain,
        queries=queries,
        structured_output=structured_output
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
