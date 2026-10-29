"""
Embed the `population_concernee` column of the contingent_evolution extraction and
upload it to a dedicated Qdrant collection, ready for downstream UMAP topic analysis.

Pipeline (mirrors notebooks/visualisation_evo_contingent.ipynb "treat_all_years"):
    - Read contingent_evolution + contingent_old + metadata parquet shards from S3
    - Global cleaning (cross-check vs old extraction, drop anonymised/recall/other-text)
    - Develop (explode) the `contingents` list into one row per contingent
    - Annualise monthly contingents
    - Join document title/date from metadata
    - Embed the `population_concernee` text (deduplicated across identical strings)
    - Upsert one point per row with all the metadata we can carry
"""
import argparse
import asyncio
import json
import logging
import math
import random
import uuid
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from qdrant_client import models
from tqdm import tqdm

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
for noisy in ("httpx", "httpx2", "httpcore", "qdrant_client", "openai", "s3fs", "fsspec"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

EMBED_MAX_RETRIES = 6
UPSERT_MAX_RETRIES = 4
MAX_PARALLEL_UPSERTS = 4
UPSERT_CHUNK = 64
CHECKPOINT_BASE = f"{config.EXTRACTION_BASE}/contingent_evolution/_umap_checkpoints"
DEFAULT_YEARS = "2018-2025"

# Payload index fields (keyword) useful for filtering in Qdrant
PAYLOAD_INDEX_FIELDS = ("metadata.reference", "metadata.anneeSignature", "metadata.type_substitution")


# --------------------------------------------------------------------------- #
# Metadata cleaning for JSON payloads
# --------------------------------------------------------------------------- #
def to_python(value: Any) -> Any:
    """Convert pandas/numpy values to JSON-serialisable Python (NaN/NaT -> None)."""
    if isinstance(value, np.ndarray):
        return [to_python(v) for v in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [to_python(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_python(v) for k, v in value.items()}
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    return value


# --------------------------------------------------------------------------- #
# Qdrant
# --------------------------------------------------------------------------- #
def ensure_collection(collection_name: str) -> None:
    """Create the collection (and payload indexes) if it does not exist."""
    client = config.qdrant_client
    if not client.collection_exists(collection_name=collection_name):
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=config.EMBEDDING_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"Collection {collection_name} created.")
        except Exception as e:  # already created concurrently
            if not client.collection_exists(collection_name=collection_name):
                raise
            logger.info(f"Collection {collection_name} created concurrently ({e}).")

    for field in PAYLOAD_INDEX_FIELDS:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # index already exists


# --------------------------------------------------------------------------- #
# Data loading + treatment (replicates the notebook's treat_all_years)
# --------------------------------------------------------------------------- #
def parse_years(value: str) -> list[int]:
    """Accept a range ('2018-2025') or a comma list ('2018,2020,2021')."""
    value = value.strip()
    if "-" in value:
        lo, hi = value.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(y) for y in value.split(",") if y.strip()]


def fetch_year(year: int):
    """Fetch (extraction_old, extraction_evo, metadata) for one year."""
    extraction_evo = pd.read_parquet(
        f"{config.EXTRACTION_BASE}/contingent_evolution/year={year}/extraction_{year}.parquet",
        filesystem=config.fs,
    )
    extraction = pd.read_parquet(
        f"{config.EXTRACTION_BASE}/contingent_old/year={year}/extraction_{year}.parquet",
        filesystem=config.fs,
    )
    metadata = pd.read_parquet(
        f"{config.METADATA_BASE}/acco_metadata_{year}.parquet",
        filesystem=config.fs,
    )
    metadata = metadata[["reference", "title", "dateSignature"]]
    return extraction, extraction_evo, metadata


def develop_contingents(df: pd.DataFrame) -> pd.DataFrame:
    """Explode the `contingents` list column into one row per contingent."""
    df = df.copy()
    if isinstance(df.dtypes["contingents"], pd.StringDtype):
        df["contingents"] = [json.loads(c) for c in df["contingents"].to_list()]
    df = df.explode("contingents")
    df_dict = pd.json_normalize(df["contingents"])
    return pd.concat([df.drop(columns=["contingents"]), df_dict], axis=1)


def treat_all_years(extraction: pd.DataFrame, extraction_evo: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates the notebook's `treat_all_years`: cross-check the evo extraction against
    the old extraction, drop noisy rows (other-legal-text / anonymised / recall-only),
    then develop the contingents and annualise monthly ones.
    """
    extraction = extraction[pd.col("erreur").isna()]
    extraction_evo = extraction_evo[pd.col("erreur").isna()]

    compare = extraction_evo.merge(extraction, on="reference", how="inner", suffixes=["_evo", ""])

    mask_presence = pd.col("presence_contingent_conventionnel_evo") | pd.col("presence_contingent_conventionnel")
    compare = compare[mask_presence]

    mask_mismatch = pd.col("presence_contingent_conventionnel_evo") ^ pd.col("presence_contingent_conventionnel")
    mismatches = compare[mask_mismatch]

    # 1) evo says absent while old says present -> probably a contingent taken from another legal text
    removed_other_text = mismatches.loc[pd.col("presence_contingent_conventionnel"), "reference"].to_list()

    # 2) hand-curated list of anonymised references
    with open(config.NOISY_PATH) as f:
        removed_anon = [x for x in f.read().split("\n") if x.strip()]

    # 3) evo says present but the text only recalls a contingent (not fixing it)
    removed_recall = mismatches[pd.col("presence_contingent_conventionnel_evo")]
    removed_recall = removed_recall[pd.col("justification_ou_reserve").str.contains("rappel", na=False)]
    removed_recall = removed_recall["reference"].to_list()

    removed = list(set(removed_other_text + removed_anon + removed_recall))

    extraction_evo = extraction_evo[pd.col("presence_contingent_conventionnel")]
    extraction_filtered = extraction_evo[~pd.col("reference").isin(removed)]

    developed = develop_contingents(extraction_filtered)

    # Contingent mensuel -> annuel
    developed.loc[pd.col("periode").str.contains("mensuel"), "nombre_heures_fixe"] *= 12
    developed.loc[pd.col("periode").str.contains("mensuel"), "periode"] = "annuelle (montant à chaque mois x 12)"

    return developed


def build_dataframe(years: list[int]) -> pd.DataFrame:
    """Fetch, treat and enrich the developed data for the given years."""
    olds, evos, metas = [], [], []
    for year in years:
        o, e, m = fetch_year(year)
        o["anneeSignature"] = year
        e["anneeSignature"] = year
        olds.append(o)
        evos.append(e)
        metas.append(m)
        logger.info(f"[{year}] evo={len(e)} old={len(o)} metadata={len(m)}")

    extraction = pd.concat(olds, ignore_index=True)
    extraction_evo = pd.concat(evos, ignore_index=True)
    metadata = pd.concat(metas, ignore_index=True).drop_duplicates(subset="reference")

    developed = treat_all_years(extraction, extraction_evo)
    logger.info(f"Developed rows after treatment: {len(developed)}")

    developed = developed.merge(metadata, on="reference", how="left")

    # Drop rows without any text to embed
    developed = developed[
        pd.col("population_concernee").notna()
        & (pd.col("population_concernee").astype(str).str.strip() != "")
    ].reset_index(drop=True)

    # Stable per-row sub-index (a reference can hold several contingents)
    developed["_sub"] = developed.groupby("reference").cumcount()
    return developed


# --------------------------------------------------------------------------- #
# Embedding (deduplicated) + upsert
# --------------------------------------------------------------------------- #
async def _embed_batch(texts: list[str], sem: asyncio.Semaphore) -> list[list[float]]:
    async with sem:
        for attempt in range(EMBED_MAX_RETRIES):
            try:
                response = await config.llm_client.embeddings.create(
                    model=config.EMBEDDING_MODEL,
                    input=texts,
                )
                data = sorted(response.data, key=lambda d: d.index)
                return [d.embedding for d in data]
            except Exception as e:
                if attempt == EMBED_MAX_RETRIES - 1:
                    raise
                wait = min(2 ** attempt * 2, 60) + random.random()
                logger.warning(f"Embedding failed ({type(e).__name__}: {e}); retry in {wait:.0f}s")
                await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


async def embed_unique(texts: list[str], batch_size: int, concurrency: int, desc: str) -> list[list[float]]:
    """Embed a list of texts (order preserved), bounded by a semaphore with backoff."""
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    sem = asyncio.Semaphore(concurrency)
    results: list[list[list[float]]] = [None] * len(batches)

    async def worker(i: int, batch: list[str]):
        results[i] = await _embed_batch(batch, sem)

    tasks = [asyncio.create_task(worker(i, b)) for i, b in enumerate(batches)]
    try:
        with tqdm(total=len(texts), desc=desc, unit="text") as pbar:
            for fut in asyncio.as_completed(tasks):
                await fut
                pbar.update(len(batches))
    except BaseException:
        for t in tasks:
            t.cancel()
        raise
    return [v for batch in results for v in batch]


def build_point(point_id: str, text: str, meta: dict) -> models.PointStruct:
    return models.PointStruct(
        id=point_id,
        vector=meta["__vector__"],
        payload={
            "page_content": text,
            "metadata": {k: to_python(v) for k, v in meta.items() if k != "__vector__"},
        },
    )


async def upsert_points(collection: str, points: list[models.PointStruct], desc: str) -> int:
    sem = asyncio.Semaphore(MAX_PARALLEL_UPSERTS)
    chunks = [points[i:i + UPSERT_CHUNK] for i in range(0, len(points), UPSERT_CHUNK)]

    async def worker(chunk: list[models.PointStruct]):
        async with sem:
            for attempt in range(UPSERT_MAX_RETRIES):
                try:
                    await asyncio.to_thread(
                        config.qdrant_client.upsert,
                        collection_name=collection,
                        points=chunk,
                        wait=True,
                    )
                    return
                except Exception as e:
                    if attempt == UPSERT_MAX_RETRIES - 1:
                        raise
                    wait = min(2 ** attempt * 3, 60) + random.random()
                    logger.warning(f"Upsert failed ({type(e).__name__}: {e}); retry in {wait:.0f}s")
                    await asyncio.sleep(wait)

    tasks = [asyncio.create_task(worker(c)) for c in chunks]
    try:
        for fut in asyncio.as_completed(tasks):
            await fut
    except BaseException:
        for t in tasks:
            t.cancel()
        raise
    return len(points)


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def checkpoint_path(collection: str, year: int) -> str:
    return f"{CHECKPOINT_BASE}/{collection}/year={year}.done"


def is_done(path: str) -> bool:
    try:
        return config.fs.exists(path)
    except Exception:
        return False


def mark_done(path: str, n: int) -> None:
    try:
        with config.fs.open(path, "w") as f:
            f.write(str(n))
    except Exception as e:
        logger.warning(f"Could not write checkpoint {path}: {e}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
META_FIELDS = [
    "reference",
    "anneeSignature",
    "n_chunks",
    "presence_contingent_conventionnel",
    "nombre_heures_fixe",
    "type_substitution",
    "periode",
    "nombre_heures_avant_accord",
    "citation_exacte",
    "justification_ou_reserve",
    "title",
    "dateSignature",
]


async def run(args) -> None:
    collection = args.collection
    years = parse_years(args.years)
    if args.limit:
        years = years[: args.limit]
    logger.info(f"Years: {years} -> collection '{collection}'")

    ensure_collection(collection)

    developed = build_dataframe(years)
    if developed.empty:
        logger.warning("No rows to process.")
        return

    unique_texts = developed["population_concernee"].astype(str).str.strip().unique().tolist()
    logger.info(f"Embedding {len(unique_texts)} unique texts (out of {len(developed)} rows)...")
    vectors = await embed_unique(
        unique_texts,
        batch_size=args.embed_batch_size,
        concurrency=args.embed_concurrency,
        desc="embedding",
    )
    vec_by_text = dict(zip(unique_texts, vectors))
    developed["__vector__"] = developed["population_concernee"].astype(str).str.strip().map(vec_by_text)

    grand_total = 0
    for year in years:
        sub = developed[pd.col("anneeSignature") == year].copy()
        marker = checkpoint_path(collection, year)
        if not args.force and is_done(marker):
            logger.info(f"[{year}] already done, skipping")
            continue
        if sub.empty:
            mark_done(marker, 0)
            continue

        points = []
        for _, row in sub.iterrows():
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"pop#{row['reference']}#{row['_sub']}"))
            meta = {k: row.get(k) for k in META_FIELDS}
            meta["__vector__"] = row["__vector__"]
            points.append(build_point(point_id, str(row["population_concernee"]), meta))

        n = await upsert_points(collection, points, desc=f"upsert {year}")
        mark_done(marker, n)
        grand_total += n
        logger.info(f"[{year}] upserted {n} points")

    logger.info(f"Done. {grand_total} points upserted into '{collection}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Embed population_concernee from the contingent_evolution extraction and upsert to Qdrant."
    )
    parser.add_argument("--years", type=str, default=DEFAULT_YEARS,
                        help="Range ('2018-2025') or comma list ('2018,2020').")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process the first N years (for testing).")
    parser.add_argument("--collection", type=str, default=config.COLLECTION,
                        help="Qdrant collection name.")
    parser.add_argument("--embed_batch_size", type=int, default=config.EMBED_BATCH_SIZE,
                        help="Number of texts per embedding request.")
    parser.add_argument("--embed_concurrency", type=int, default=config.EMBED_CONCURRENCY,
                        help="Max simultaneous embedding requests.")
    parser.add_argument("--force", action="store_true",
                        help="Ignore checkpoints and reprocess every year.")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
