"""
Chunking + embedding + export to Qdrant, ready for RAG LangChain.

Script originaly written by Gemini for Qdrant-langchain interface and rag chunking.
General pipeline done by me, then refactored by Claude (see commit history) for faster computations
on databases of approx. 50,000 lines each :

    - Treatment shard by shard (streaming)
    - Async embeddings by batch (semaphore + retry and potential backoff), then upsert in Qdrant
    - Deterministic ids (uuid5 de doc_id + index du chunk): restarting jobs do not provide duplacates
    - Checkpoint by shard on S3 (file .done)
    - Compatible payload with langchain_qdrant.QdrantVectorStore
    - Cleaned metadata (NaN, numpy, dates) for JSON.
    - Une collection partagée par défaut (filtrage par `metadata.data_folder` / année), au lieu
      d'une collection par base.
"""
import argparse
import asyncio
import hashlib
import logging
import math
import os
import random
import uuid
from datetime import date, datetime
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from qdrant_client import models
from tqdm import tqdm

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
for noisy in ("httpx", "httpx2", "httpcore", "qdrant_client", "openai", "s3fs", "fsspec"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

DEFAULT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "accords")
DEFAULT_EMBED_BATCH_SIZE = getattr(config, "EMBED_BATCH_SIZE", 64)
MIN_CHUNK_CHARS = 30
EMBED_MAX_RETRIES = 6
MAX_PARALLEL_UPSERTS = 4

# Colonnes inutiles pour le RAG
COLUMNS_TO_DROP = ["legalStatus", "startDate", "endDate", "nature"]


# --------------------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------------------
def parse_theme_filter(value: str):
    """
    Either 'all' or a list of themes, each quoted with '"' and separated by commas
    (e.g. '"theme1","theme2"').
    """
    if value.strip().lower() == "all":
        return "all"
    themes = [str(m.strip('"')) for m in value.split('","')]
    if not themes:
        raise argparse.ArgumentTypeError(
            """Themes must be 'all' or themes delimited by '"' and separated by commas."""
        )
    return themes


def themes_tag(themes: str | list[str]) -> str:
    """Short, filesystem-safe tag identifying the theme filter (used for checkpoints)."""
    if themes == "all":
        return "all"
    return hashlib.sha1("|".join(sorted(themes)).encode()).hexdigest()[:10]


# --------------------------------------------------------------------------------------
# Qdrant
# --------------------------------------------------------------------------------------
def ensure_collection(collection_name: str) -> None:
    """
    Create the collection (and useful payload indexes) if it does not exist.
    Tolerates the race where several parallel jobs create it at the same time.
    Payload layout is the one expected by langchain_qdrant (metadata.* keys).
    """
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
        except Exception as e:  # already created by a parallel job
            if not client.collection_exists(collection_name=collection_name):
                raise
            logger.info(f"Collection {collection_name} created concurrently ({e}).")

    for field in ("metadata.data_folder", "metadata.doc_id"):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # index already exists


# --------------------------------------------------------------------------------------
# Lecture / nettoyage des données
# --------------------------------------------------------------------------------------
def list_shards(path: str) -> list[str]:
    """List parquet shards (sorted, deterministic order) under an S3 path."""
    files = [f for f in config.fs.ls(path) if f.endswith(".parquet")]
    return sorted(files)


def read_shard(file: str) -> pd.DataFrame:
    return pd.read_parquet("s3://" + file, filesystem=config.fs)


def clean_database(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unnecessary columns, rename columns of interest.
    The original `id` is kept as `doc_id` (needed for deterministic chunk ids).
    """
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")
    df["dateDiffusion"] = pd.to_datetime(df["dateDiffusion"]).dt.strftime("%Y-%m-%d")
    df["dateSignature"] = df["dateSignature"].dt.strftime("%Y-%m-%d")
    df = df.rename(
        columns={
            "id": "doc_id",
            "YYYY": "anneeSignature",
            "MM": "moisSignature",
        }
    )
    # Drop documents without any text: nothing to embed
    df = df[df["markdown"].notna() & (df["markdown"].astype(str).str.strip() != "")]
    return df.reset_index(drop=True)


def filter_database_by_theme(df: pd.DataFrame, themes: list[str] | str) -> pd.DataFrame:
    """
    Keep data only belonging to certain themes (boolean columns, OR across themes).
    """
    if isinstance(themes, str):
        if themes == "all":
            return df
        raise TypeError(f"themes {themes} is of type str but not equal to all")
    missing = [t for t in themes if t not in df.columns]
    if missing:
        raise KeyError(f"Unknown theme column(s): {missing}")
    return df[df[themes].any(axis=1)]


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


def split_markdown_metadata(df: pd.DataFrame) -> tuple[list[str], list[dict[str, Any]]]:
    markdowns = df["markdown"].astype(str).to_list()
    records = df.drop(columns=["markdown"]).to_dict(orient="records")
    metadata = [{k: to_python(v) for k, v in rec.items()} for rec in records]
    return markdowns, metadata


# --------------------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=4)
def get_splitters(chunk_size: int, chunk_overlap: int):
    """Build the splitters once (not once per document)."""
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ],
        strip_headers=False,
    )
    # Ordered separators: paragraph, end of sentence, list items, words, characters
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", ".\n", ". ", "\n- ", "\n1. ", " ", ""],
    )
    return markdown_splitter, recursive_splitter


def chunk_markdown_with_headers(
    markdown_content: str,
    extra_metadata: dict[str, Any],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """
    Chunk markdown, preserving global structure when possible (header splitting).
    If parts are too big, split into paragraphs, then sentences, then lists, then words.
    Each chunk is prefixed with its header breadcrumb to keep context at retrieval time.
    """
    markdown_splitter, recursive_splitter = get_splitters(chunk_size, chunk_overlap)

    chunks: list[Document] = []
    for doc in markdown_splitter.split_text(markdown_content):
        for sub_doc in recursive_splitter.split_documents([doc]):
            text = sub_doc.page_content.strip()
            if len(text) < MIN_CHUNK_CHARS:
                continue

            breadcrumb = " > ".join(
                v for k, v in sub_doc.metadata.items() if k.startswith("Header_")
            )
            content = f"[Contexte: {breadcrumb}]\n{text}" if breadcrumb else text

            chunks.append(
                Document(
                    page_content=content,
                    metadata={**sub_doc.metadata, **extra_metadata},
                )
            )

    # Position of each chunk in its document
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["n_chunks"] = len(chunks)
    return chunks


def make_point_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic point id -> re-running a job overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}#{chunk_index}"))


def chunk_dataframe(df: pd.DataFrame, data_folder: str, shard_name: str) -> list[tuple[str, Document]]:
    """Chunk every document of a shard. Returns (point_id, Document) pairs."""
    markdowns, metadata = split_markdown_metadata(df)
    out = []
    for md, meta in zip(markdowns, metadata):
        doc_id = str(meta.get("doc_id") or hashlib.sha1(md.encode()).hexdigest())
        meta["doc_id"] = doc_id
        meta["data_folder"] = data_folder
        meta["shard"] = shard_name
        for chunk in chunk_markdown_with_headers(md, meta):
            out.append((make_point_id(doc_id, chunk.metadata["chunk_index"]), chunk))
    return out


# --------------------------------------------------------------------------------------
# Embedding + upsert (asynchrone)
# --------------------------------------------------------------------------------------
async def embed_chunks(texts: list[str], semaphore: asyncio.Semaphore) -> list[list[float]]:
    """
    Embed one batch of texts, bounded by the semaphore, with exponential backoff on errors
    (rate limits, timeouts...). Results are re-ordered by `index` to be safe.
    """
    async with semaphore:
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


async def embed_and_upsert_batch(
    batch: list[tuple[str, Document]],
    collection: str,
    embed_sem: asyncio.Semaphore,
    upsert_sem: asyncio.Semaphore,
) -> int:
    vectors = await embed_chunks([doc.page_content for _, doc in batch], embed_sem)
    points = [
        models.PointStruct(
            id=point_id,
            vector=vector,
            # Layout expected by langchain_qdrant.QdrantVectorStore
            payload={"page_content": doc.page_content, "metadata": doc.metadata},
        )
        for (point_id, doc), vector in zip(batch, vectors)
    ]
    async with upsert_sem:
        await asyncio.to_thread(
            config.qdrant_client.upsert,
            collection_name=collection,
            points=points,
            wait=True,
        )
    return len(points)


async def embed_and_upsert(
    chunks: list[tuple[str, Document]],
    collection: str,
    batch_size: int,
    embed_concurrency: int,
    desc: str,
) -> int:
    """Embed and upsert all chunks of a shard; stops everything on first failure."""
    embed_sem = asyncio.Semaphore(embed_concurrency)
    upsert_sem = asyncio.Semaphore(MAX_PARALLEL_UPSERTS)
    tasks = [
        asyncio.create_task(
            embed_and_upsert_batch(chunks[i:i + batch_size], collection, embed_sem, upsert_sem)
        )
        for i in range(0, len(chunks), batch_size)
    ]
    total = 0
    try:
        with tqdm(total=len(chunks), desc=desc, unit="chunk") as pbar:
            for fut in asyncio.as_completed(tasks):
                n = await fut
                total += n
                pbar.update(n)
    except BaseException:
        for t in tasks:
            t.cancel()
        raise
    return total


# --------------------------------------------------------------------------------------
# Checkpoints (un marqueur .done par shard)
# --------------------------------------------------------------------------------------
def checkpoint_path(collection: str, data_folder: str, themes: str | list[str], shard: str) -> str:
    base = os.environ.get("CHECKPOINT_PATH", config.FULL_DATA_PATH.rstrip("/") + "_checkpoints")
    name = os.path.basename(shard)
    return f"{base.rstrip('/')}/{collection}/{data_folder}/{themes_tag(themes)}/{name}.done"


def is_done(path: str) -> bool:
    try:
        return config.fs.exists(path)
    except Exception:
        return False


def mark_done(path: str, n_chunks: int) -> None:
    try:
        with config.fs.open(path, "w") as f:
            f.write(str(n_chunks))
    except Exception as e:
        logger.warning(f"Could not write checkpoint {path}: {e} (upserts stay idempotent anyway)")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
async def run(args) -> None:
    collection = args.collection
    ensure_collection(collection)

    data_path = config.FULL_DATA_PATH + args.data_folder
    shards = list_shards(data_path)
    if args.limit_shards:
        shards = shards[:args.limit_shards]
    logger.info(f"{len(shards)} shard(s) found in {data_path} -> collection '{collection}'")

    grand_total = 0
    for idx, shard in enumerate(shards, start=1):
        shard_name = os.path.basename(shard)
        marker = checkpoint_path(collection, args.data_folder, args.themes, shard)
        if not args.force and is_done(marker):
            logger.info(f"[{idx}/{len(shards)}] {shard_name}: already done, skipping")
            continue

        df = read_shard(shard)
        df = clean_database(df)
        df = filter_database_by_theme(df, args.themes)
        if df.empty:
            logger.info(f"[{idx}/{len(shards)}] {shard_name}: no rows after filtering")
            mark_done(marker, 0)
            continue

        chunks = chunk_dataframe(df, args.data_folder, shard_name)
        logger.info(f"[{idx}/{len(shards)}] {shard_name}: {len(df)} docs -> {len(chunks)} chunks")
        del df

        n = await embed_and_upsert(
            chunks,
            collection=collection,
            batch_size=args.embed_batch_size,
            embed_concurrency=args.embed_concurrency,
            desc=shard_name,
        )
        mark_done(marker, n)
        grand_total += n

    logger.info(f"Everything is finished! {grand_total} chunks upserted into '{collection}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Chunk ACCO markdown shards, embed them and upsert them into Qdrant"
    )
    parser.add_argument(
        "--data_folder", type=str, required=True,
        help="Name of the data folder containing all the shards (e.g. acco_data_2022)",
    )
    parser.add_argument(
        "--themes", type=parse_theme_filter, default="all",
        help="""Specific themes, quoted ("") and separated by commas, or 'all'""",
    )
    parser.add_argument(
        "--collection", type=str, default=DEFAULT_COLLECTION,
        help="Qdrant collection (shared by all databases by default; filter on metadata.data_folder).",
    )
    parser.add_argument(
        "--embed_batch_size", type=int, default=DEFAULT_EMBED_BATCH_SIZE,
        help="Number of chunks per embedding request.",
    )
    parser.add_argument(
        "--embed_concurrency", type=int, default=config.EMBED_CONCURRENCY,
        help="Max simultaneous embedding requests.",
    )
    parser.add_argument(
        "--limit_shards", type=int, default=0,
        help="Only process the first N shards (for testing).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore checkpoints and reprocess every shard.",
    )
    args = parser.parse_args()
    logger.info(args.themes)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
