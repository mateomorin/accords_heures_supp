"""
RAG extraction of the "contingent d'heures supplémentaires" by year of signature.

For each anneeSignature:
  1. list all distinct metadata.reference for the year;
  2. async for each reference : filtered retrieval (top-k chunks) + structured LLM call;
  3. results are written by shards on S3 (retry available after Argo crash/retry),
     then merged into a unique parquet by year.

Exzmples :
  python extract_contingent.py --years 2022
  python extract_contingent.py                      # 2017 → 2025
  python extract_contingent.py --years 2022 --references REF1 REF2   # debug
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from qdrant_client.http import models
from tqdm.asyncio import tqdm_asyncio

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
for noisy in ("httpx", "httpx2", "httpcore", "qdrant_client", "openai", "s3fs", "fsspec"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

SCRIPT_DIR = Path(__file__).parent


# --------------------------------------------------------------------------- #
# Output scheme
# --------------------------------------------------------------------------- #
class DetailContingent(BaseModel):
    categorie_ou_population: str | None = Field(
        default=None,
        description="Population concernée si le contingent est spécifique (ex: 'Non-cadres', 'Salariés en modulation', 'Tous les salariés').",
    )
    nombre_heures: int = Field(
        description="Nombre d'heures du contingent d'heures supplémentaires fixé par l'accord."
    )
    periode: str | None = Field(
        default="annuel",
        description="Période de référence du contingent (ex: 'annuel', 'par exercice').",
    )


class ExtractionContingentHeuresSup(BaseModel):
    presence_contingent_conventionnel: bool = Field(
        description=(
            "True uniquement si le document fixe un contingent d'heures supplémentaires "
            "propre à l'entreprise. False si le document est muet ou ne fait que citer le contingent légal de référence."
        )
    )
    contingents: list[DetailContingent] = Field(
        default_factory=list,
        description="Liste des contingents d'heures supplémentaires définis spécifiquement par l'accord.",
    )
    citation_exacte: str | None = Field(
        default=None,
        description="Extrait exact du texte traitant du contingent d'heures supplémentaires. Obligatoire si un contingent est trouvé.",
    )
    justification_ou_reserve: str | None = Field(
        default=None,
        description="Remarques éventuelles (ex: rappel du droit commun, contingent variable, proratisation).",
    )


# --------------------------------------------------------------------------- #
# Prompts / query
# --------------------------------------------------------------------------- #
def create_prompt() -> ChatPromptTemplate:
    system_prompt = (SCRIPT_DIR / "system_prompt.txt").read_text(encoding="utf-8")
    user_prompt = (SCRIPT_DIR / "user_prompt.txt").read_text(encoding="utf-8")
    return ChatPromptTemplate.from_messages([("system", system_prompt), ("human", user_prompt)])


def create_query() -> str:
    return (SCRIPT_DIR / "query.txt").read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------- #
# Qdrant : filters and reference listing
# --------------------------------------------------------------------------- #
def reference_filter(reference) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key=config.REF_KEY, match=models.MatchValue(value=reference))]
    )


def year_filter(year: int) -> models.Filter:
    return models.Filter(
        should=[
            models.FieldCondition(key=config.YEAR_KEY, match=models.MatchValue(value=str(year))),
        ]
    )


async def list_references(year: int, limit: int = 10000) -> list:
    """
    List all distinct references after QDrant Facet.
    """
    flt = year_filter(year)
    response = await asyncio.to_thread(
            config.qdrant_client.facet,
            key=config.REF_KEY,
            collection_name=config.QDRANT_COLLECTION,
            facet_filter=flt,
            limit=limit,
        )

    hits = response.hits

    refs = [hit.value for hit in hits]
    n_points = sum([hit.count for hit in hits])

    logger.info(f"[{year}] {n_points} chunks scanned → {len(refs)} distinct references")
    return sorted(refs, key=str)


# --------------------------------------------------------------------------- #
# Extraction for one reference
# --------------------------------------------------------------------------- #
async def extract_one(reference, year, chain, retriever_factory, query, semaphore) -> dict:
    row = {
        "reference": reference,
        "anneeSignature": year,
        "presence_contingent_conventionnel": None,
        "contingents": None,
        "citation_exacte": None,
        "justification_ou_reserve": None,
        "n_chunks": 0,
        "erreur": None,
    }
    async with semaphore:
        try:
            retriever = retriever_factory(reference)
            docs = await retriever.ainvoke(query)
            row["n_chunks"] = len(docs)

            if not docs:
                row.update(
                    presence_contingent_conventionnel=False,
                    contingents="[]",
                    justification_ou_reserve="Aucun chunk retrouvé pour cette reference.",
                )
                return row

            context_text = "\n\n".join(d.page_content for d in docs)
            res = await chain.ainvoke({"context": context_text})
            if res is None:  # with_structured_output might return None if parsing fails
                raise ValueError("Sortie structurée vide (parsing LLM échoué)")

            row.update(
                presence_contingent_conventionnel=res.presence_contingent_conventionnel,
                contingents=json.dumps([c.model_dump() for c in res.contingents], ensure_ascii=False),
                citation_exacte=res.citation_exacte,
                justification_ou_reserve=res.justification_ou_reserve,
            )
        except Exception as e:
            logger.error(f"[{year}] Error with {reference}: {e!r}")
            row["erreur"] = repr(e)[:500]
    return row


# --------------------------------------------------------------------------- #
# Extraction for one year
# --------------------------------------------------------------------------- #
def _read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path, filesystem=config.fs)


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, filesystem=config.fs, index=False)


async def process_year(year: int, args, chain, query) -> int:
    """Returns the number of references that failed."""
    base = f"{config.S3_OUTPUT_PATH.rstrip('/')}/year={year}"
    final_path = f"{base}/extraction_{year}.parquet"

    if args.skip_existing and config.fs.exists(final_path):
        logger.info(f"[{year}] {final_path} already exists, skip.")
        return 0

    refs = args.references or await list_references(year)
    if args.limit:
        refs = refs[: args.limit]
    if not refs:
        logger.warning(f"[{year}] No reference found.")
        return 0

    def retriever_factory(reference):
        return config.vector_store.as_retriever(
            search_kwargs={"k": args.top_k, "filter": reference_filter(reference)}
        )

    semaphore = asyncio.Semaphore(args.concurrency)
    shard_dfs = []
    n_shards = (len(refs) + args.shard_size - 1) // args.shard_size

    for s in range(n_shards):
        batch = refs[s * args.shard_size: (s + 1) * args.shard_size]
        shard_path = f"{base}/shards/shard_{s:05d}.parquet"

        # Checkpoint: keep correct lines from the existing shard and recompute only errors/missing items
        kept = pd.DataFrame()
        if config.fs.exists(shard_path):
            prev = _read_parquet(shard_path)
            kept = prev[prev["erreur"].isna()]
            batch = [r for r in batch if r not in set(kept["reference"])]
            if not batch:
                logger.info(f"[{year}] shard {s + 1}/{n_shards} already full.")
                shard_dfs.append(kept)
                continue

        new_rows = await tqdm_asyncio.gather(
            *[extract_one(r, year, chain, retriever_factory, query, semaphore) for r in batch],
            desc=f"[{year}] shard {s + 1}/{n_shards}",
        )
        shard_df = pd.concat([kept, pd.DataFrame(new_rows)], ignore_index=True)
        _write_parquet(shard_df, shard_path)  # écrit immédiatement : un crash ne perd qu'un shard
        shard_dfs.append(shard_df)

    full = pd.concat(shard_dfs, ignore_index=True)
    n_err = int(full["erreur"].notna().sum())
    n_pos = int((full["presence_contingent_conventionnel"].eq(True)).sum())
    logger.info(f"[{year}] {len(full)} references | {n_pos} with contingent conventionnel | {n_err} errors")

    if n_err == 0:
        _write_parquet(full, final_path)
        logger.info(f"[{year}] Final result exported to: {final_path}")
    else:
        logger.warning(f"[{year}] Final result has not been exported ({n_err} errors).")
    return n_err


async def main_async(args) -> int:
    structured_llm = config.llm.with_structured_output(ExtractionContingentHeuresSup)
    chain = (create_prompt() | structured_llm).with_retry(
        stop_after_attempt=args.llm_retries, wait_exponential_jitter=True
    )
    query = create_query()

    total_errors = 0
    for year in args.years:  # années en séquence ; la concurrence est portée par le semaphore
        total_errors += await process_year(year, args, chain, query)
    return total_errors


def main():
    parser = argparse.ArgumentParser(description="Extraction du contingent d'heures supp. des accords ACCO, par année.")
    parser.add_argument("--years", nargs="+", type=int, default=config.DEFAULT_YEARS,
                        help="Signature years to treat (default: 2017 to 2025).")
    parser.add_argument("--references", nargs="+", type=str, default=[],
                        help="(debug) Only tackles special references.")
    parser.add_argument("--limit", type=int, default=0, help="(debug) Only treat the first N references.")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max number of (retrieval + LLM) per process.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks per document.")
    parser.add_argument("--shard_size", type=int, default=500,
                        help="Number of references per shard to save on S3.")
    parser.add_argument("--llm_retries", type=int, default=4, help="Number of LLM API call attemps (potential backoff).")
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True,
                        help="Ignore years where parquet file already exists.")
    args = parser.parse_args()

    n_errors = asyncio.run(main_async(args))
    if n_errors:
        logger.error(f"{n_errors} references with errors → exit 1 (Argo will start the pod again).")
        sys.exit(1)


if __name__ == "__main__":
    main()
