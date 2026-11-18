"""
Chunking + embedding + export of the v2 structural parts into Qdrant.

Input: `s3://mateomorin/legifrance/rag/v2/shards` (one row per document, `parts_json`
holds the parts produced by `src/preprocessing`).

Each part is embedded as a whole unless it is longer than `CHUNK_SIZE`; long parts are
split and every chunk keeps the part breadcrumb as a prefix. Tables are attached to the
part that carries them and are part of the embedded text: majoration grids live there.

The run is streaming (one shard at a time), idempotent (uuid5 point ids) and resumable
(one `.done` marker per shard on S3). Payload layout is the one expected by
`langchain_qdrant.QdrantVectorStore`.
"""

import argparse
import asyncio
import json
import logging
import os
import random
import uuid
from typing import Any

import config
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
for noisy in ("httpx", "httpx2", "httpcore", "qdrant_client", "openai", "s3fs", "fsspec"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

EMBED_MAX_RETRIES = 6


# --------------------------------------------------------------------------------------
# Qdrant
# --------------------------------------------------------------------------------------
def ensure_collection(collection_name: str) -> None:
    """
    Create the collection (+ payload indexes) if it does not exist.

    Vectors are quantized to int8 and kept in RAM: ~1M parts x 4096 dims would otherwise
    cost ~16 GB of float32. Tolerates several jobs creating it concurrently.
    """
    client = config.qdrant_client
    if not client.collection_exists(collection_name=collection_name):
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=config.EMBEDDING_SIZE,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                ),
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                ),
            )
            logger.info(f"Collection {collection_name} created.")
        except Exception as e:
            if not client.collection_exists(collection_name=collection_name):
                raise
            logger.info(f"Collection {collection_name} created concurrently ({e}).")

    for field in ("metadata.reference", "metadata.anneeSignature", "metadata.section_id"):
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


# --------------------------------------------------------------------------------------
# Parts -> texts
# --------------------------------------------------------------------------------------
def list_shards(path: str) -> list[str]:
    """List parquet shards under an S3 path (sorted, deterministic order)."""
    return sorted(f for f in config.fs.ls(path) if f.endswith(".parquet"))


def part_body(part: dict[str, Any]) -> str:
    """Body markdown followed by the markdown of its tables."""
    blocks = [part.get("body_md") or ""]
    blocks += [table.get("md") or "" for table in part.get("tables") or []]
    return "\n\n".join(block.strip() for block in blocks if block and block.strip())


def part_header(part: dict[str, Any]) -> str:
    """Breadcrumb + title, repeated on every chunk of the part."""
    path = part.get("path") or ""
    title = part.get("title") or ""
    return "\n".join(part for part in (path, title) if part)


def part_metadata(row: dict[str, Any], part: dict[str, Any], shard: str) -> dict[str, Any]:
    year = row.get("anneeSignature")
    year_value = None if year is None or pd.isna(year) else str(int(year))
    return {
        "reference": row["reference"],
        # Stored as a string: the extraction filters on str(year) (Qdrant keyword match)
        "anneeSignature": year_value,
        "cid": row.get("cid"),
        "doc_type": row.get("doc_type"),
        "shard": shard,
        "section_id": part["section_id"],
        "section_order": part["section_order"],
        "level": part["level"],
        "number": part.get("number"),
        "title": part.get("title"),
        "path": part.get("path"),
        "parent_id": part.get("parent_id"),
        "origin": part.get("origin"),
    }


def make_point_id(reference: str, section_id: str, chunk_index: int) -> str:
    """Deterministic point id: re-running a job overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{reference}#{section_id}#{chunk_index}"))


def part_documents(
    row: dict[str, Any],
    part: dict[str, Any],
    shard: str,
    splitter: RecursiveCharacterTextSplitter,
) -> list[tuple[str, Document]]:
    """Chunks of one part. Container parts (empty body) produce no point."""
    body = part_body(part)
    if not body:
        return []

    chunks = splitter.split_text(body) if len(body) > config.CHUNK_SIZE else [body]
    header = part_header(part)
    metadata = part_metadata(row, part, shard)

    documents = []
    for index, chunk in enumerate(chunks):
        text = f"{header}\n\n{chunk}" if header else chunk
        if len(text.strip()) < config.MIN_CHUNK_CHARS:
            continue
        documents.append(
            (
                make_point_id(row["reference"], part["section_id"], index),
                Document(
                    page_content=text,
                    metadata={**metadata, "chunk_index": index, "n_chunks": len(chunks)},
                ),
            )
        )
    return documents


def shard_documents(df: pd.DataFrame, shard: str) -> list[tuple[str, Document]]:
    """Chunk every part of every document of a shard."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", ".\n", ". ", "\n- ", "\n1. ", " ", ""],
    )
    out: list[tuple[str, Document]] = []
    for row in df.to_dict(orient="records"):
        parts = json.loads(row["parts_json"])
        for part in parts:
            out.extend(part_documents(row, part, shard, splitter))
    return out


# --------------------------------------------------------------------------------------
# Embedding + upsert
# --------------------------------------------------------------------------------------
async def embed_texts(texts: list[str], semaphore: asyncio.Semaphore) -> list[list[float]]:
    """Embed one batch, bounded by the semaphore, with exponential backoff."""
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
                wait = min(2**attempt * 2, 60) + random.random()
                logger.warning(f"Embedding failed ({type(e).__name__}: {e}); retry in {wait:.0f}s")
                await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


async def embed_and_upsert_batch(
    batch: list[tuple[str, Document]],
    collection: str,
    embed_sem: asyncio.Semaphore,
    upsert_sem: asyncio.Semaphore,
) -> int:
    vectors = await embed_texts([doc.page_content for _, doc in batch], embed_sem)
    points = [
        models.PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "page_content": doc.page_content,
                "metadata": {**doc.metadata, "chunk_id": str(point_id)},
            },
        )
        for (point_id, doc), vector in zip(batch, vectors, strict=True)
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
    documents: list[tuple[str, Document]],
    collection: str,
    batch_size: int,
    embed_concurrency: int,
    desc: str,
) -> int:
    """Embed and upsert all chunks of a shard; stops everything on first failure."""
    embed_sem = asyncio.Semaphore(embed_concurrency)
    upsert_sem = asyncio.Semaphore(config.MAX_PARALLEL_UPSERTS)
    tasks = [
        asyncio.create_task(
            embed_and_upsert_batch(documents[i: i + batch_size], collection, embed_sem, upsert_sem)
        )
        for i in range(0, len(documents), batch_size)
    ]
    total = 0
    try:
        with tqdm(total=len(documents), desc=desc, unit="chunk") as pbar:
            for future in asyncio.as_completed(tasks):
                n = await future
                total += n
                pbar.update(n)
    except BaseException:
        for task in tasks:
            task.cancel()
        raise
    return total


# --------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------
def checkpoint_path(collection: str, shard: str) -> str:
    return f"{config.CHECKPOINT_PATH.rstrip('/')}/{collection}/{os.path.basename(shard)}.done"


def is_done(path: str) -> bool:
    try:
        return config.fs.exists(path)
    except Exception:
        return False


def mark_done(path: str, n_chunks: int) -> None:
    try:
        with config.fs.open(path, "w") as file:
            file.write(str(n_chunks))
    except Exception as e:
        logger.warning(f"Could not write checkpoint {path}: {e} (upserts stay idempotent)")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def read_shard(path: str) -> pd.DataFrame:
    return pd.read_parquet(
        f"s3://{path}",
        filesystem=config.fs,
        columns=["reference", "anneeSignature", "cid", "doc_type", "n_parts", "parts_json"],
    )


async def run(args) -> None:
    collection = args.collection
    if not args.dry_run:
        ensure_collection(collection)

    shards = list_shards(config.PARTS_PATH)
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    logger.info(f"{len(shards)} shard(s) found in {config.PARTS_PATH} -> collection '{collection}'")

    grand_total = 0
    grand_parts = 0
    for index, shard in enumerate(shards, start=1):
        shard_name = os.path.basename(shard)
        marker = checkpoint_path(collection, shard)
        if not args.dry_run and not args.force and is_done(marker):
            logger.info(f"[{index}/{len(shards)}] {shard_name}: already done, skipping")
            continue

        df = read_shard(shard)
        if df.empty:
            logger.info(f"[{index}/{len(shards)}] {shard_name}: empty shard")
            if not args.dry_run:
                mark_done(marker, 0)
            continue

        documents = shard_documents(df, shard_name)
        n_parts = int(df["n_parts"].sum())
        logger.info(
            f"[{index}/{len(shards)}] {shard_name}: {len(df)} docs / {n_parts} parts "
            f"-> {len(documents)} chunks"
        )
        del df

        if args.dry_run:
            grand_total += len(documents)
            grand_parts += n_parts
            continue

        n = await embed_and_upsert(
            documents,
            collection=collection,
            batch_size=args.embed_batch_size,
            embed_concurrency=args.embed_concurrency,
            desc=shard_name,
        )
        mark_done(marker, n)
        grand_total += n
        grand_parts += n_parts

    if args.dry_run:
        logger.info(f"Dry run: {grand_total} chunks / {grand_parts} parts, nothing written.")
    else:
        logger.info(f"Done: {grand_total} chunks upserted into '{collection}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Chunk the v2 structural parts, embed them and upsert them into Qdrant"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=config.DEFAULT_COLLECTION,
        help="Qdrant collection receiving the part chunks.",
    )
    parser.add_argument(
        "--embed_batch_size",
        type=int,
        default=config.EMBED_BATCH_SIZE,
        help="Number of chunks per embedding request.",
    )
    parser.add_argument(
        "--embed_concurrency",
        type=int,
        default=config.EMBED_CONCURRENCY,
        help="Max simultaneous embedding requests.",
    )
    parser.add_argument(
        "--limit_shards",
        type=int,
        default=0,
        help="Only process the first N shards (for testing).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore checkpoints and reprocess every shard.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute chunk counts without embedding or writing anything.",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
