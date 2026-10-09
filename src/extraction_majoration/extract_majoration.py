"""
RAG extraction of the "majorations des heures supplémentaires" by year of signature.

For each anneeSignature:
  1. list all distinct metadata.reference for the year;
  2. async for each reference : filtered retrieval (top-k chunks) + structured LLM call;
  3. results are written by shards on S3 (retry available after Argo crash/retry),
     then merged into a unique parquet by year.
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from enum import Enum
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
class TypeMajoration(str, Enum):
    POURCENTAGE = "pourcentage"  # Ex: 25 for 25%
    COEFFICIENT = "coefficient"  # Ex: 1.25 for 125%
    FORFAIT_EURO = "forfait_euro"  # Ex: 250€


class PalierMajoration(BaseModel):
    rang_debut: int | None = Field(
        default=None,
        description="Heure de début du palier (ex: 36 pour la 36e heure, soit la 1re heure sup). Null si non précisé.",
    )
    rang_fin: int | None = Field(
        default=None,
        description="Heure de fin du palier inclus (ex: 43 pour la 43e heure). Null si le palier s'applique sans limite supérieure.",
    )
    volume_heures: int | None = Field(
        default=None,
        description="Alternative au rang : nombre d'heures concernées par ce palier (ex: 8 pour 'les 8 premières heures sup').",
    )
    taux_valeur: float = Field(
        description="Valeur numérique de la majoration. Ex: 25.0 pour 25%, 1.25 pour un coefficient 1,25, ou 10.0 pour 10€/h.",
    )
    type_majoration: TypeMajoration = Field(
        default=TypeMajoration.POURCENTAGE,
        description="Unité de la valeur numérique renseignée.",
    )
    condition_particuliere: str | None = Field(
        default=None,
        description="Condition spécifique au palier (ex: 'Uniquement pour le travail du dimanche', 'De nuit', 'Si dépassement du contingent').",
    )


class RegleMajoration(BaseModel):
    categorie_ou_population: str | None = Field(
        default=None,
        description="Population ciblée (ex: 'Tous les salariés', 'Cadres en heures', 'Ouvriers', 'Temps partiel').",
    )
    duree_hebdomadaire_de_travail: int | None = Field(
        default=None,
        description="Durée hebdomadaire de travail fixée par l'accord, en heures (par exemple, 35)"
    )
    paliers: list[PalierMajoration] = Field(
        default_factory=list,
        description="Liste ordonnée des paliers de majoration applicables pour cette population/règle.",
    )
    remplacement_repos: bool | None = Field(
        default=None,
        description="True si la majoration peut être remplacée par un repos compensateur équivalent (total ou partiel).",
    )


class ExtractionMajorationsHeuresSup(BaseModel):
    presence_regle_conventionnelle: bool = Field(
        description=(
            "True si le document définit explicitement des taux ou règles de majoration d'heures supplémentaires. "
            "False si le document est muet ou renvoie purement au Code du travail sans précision."
        )
    )
    regles_majorations: list[RegleMajoration] = Field(
        default_factory=list,
        description="Ensemble des règles de majoration identifiées par population ou cas de figure.",
    )
    citations_exactes: list[str] = Field(
        default_factory=list,
        description="Extrait(s) exact(s) du texte source mentionnant les taux de majoration.",
    )
    justification_ou_reserve: str | None = Field(
        default=None,
        description="Remarques explicatives (ex: renvoi à la convention collective nationale, dispositions temporaires).",
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
        "presence_regle_conventionnelle": None,
        "regles_majorations": None,
        "citations_exactes": None,
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
                    presence_regle_conventionnelle=False,
                    regles_majorations="[]",
                    justification_ou_reserve="Aucun chunk retrouvé pour cette reference.",
                )
                return row

            context_text = "\n\n".join(d.page_content for d in docs)
            res = await chain.ainvoke({"context": context_text})
            if res is None:  # with_structured_output might return None if parsing fails
                raise ValueError("Sortie structurée vide (parsing LLM échoué)")

            row.update(
                presence_regle_conventionnelle=res.presence_regle_conventionnelle,
                regles_majorations=json.dumps([c.model_dump() for c in res.regles_majorations], ensure_ascii=False),
                citations_exactes=res.citations_exactes,
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
    n_pos = int((full["presence_regle_conventionnelle"].eq(True)).sum())
    logger.info(f"[{year}] {len(full)} references | {n_pos} with règle conventionnelle | {n_err} errors")

    if n_err == 0:
        _write_parquet(full, final_path)
        logger.info(f"[{year}] Final result exported to: {final_path}")
    else:
        logger.warning(f"[{year}] Final result has not been exported ({n_err} errors).")
    return n_err


async def main_async(args) -> int:
    structured_llm = config.llm.with_structured_output(ExtractionMajorationsHeuresSup)
    chain = (create_prompt() | structured_llm).with_retry(
        stop_after_attempt=args.llm_retries, wait_exponential_jitter=True
    )
    query = create_query()

    total_errors = 0
    for year in args.years:  # années en séquence ; la concurrence est portée par le semaphore
        total_errors += await process_year(year, args, chain, query)
    return total_errors


def main():
    parser = argparse.ArgumentParser(description="Extraction du taux de majoration des heures supp. des accords ACCO, par année.")
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
