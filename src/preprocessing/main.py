"""Preprocess final document shards into structured RAG v2 parts (deterministic)."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import config
import pandas as pd
from structure import doc_to_row
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
for noisy in ("httpx", "httpx2", "httpcore", "s3fs", "fsspec"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def process_shard(
    input_path: str, output_path: str, limit_docs: int = 0, dry_run: bool = False
) -> int:
    """Process one input shard and write one output shard."""
    df = pd.read_parquet(input_path, filesystem=config.fs)
    if limit_docs:
        df = df.head(limit_docs)

    logger.info("Processing %s: %d document(s)", input_path, len(df))

    all_rows: list[dict] = []
    for _, row in tqdm(list(df.iterrows()), desc=input_path, leave=False):
        doc = json.loads(row["doc_json"])
        metadata = {
            "cid": row["cid"],
            "reference": row["reference"],
            "dateSignature": row.get("dateSignature"),
            "doc_type": row.get("doc_type"),
            "format": row.get("format"),
        }
        all_rows.append(doc_to_row(doc, metadata=metadata))

    out_df = pd.DataFrame(all_rows)
    if dry_run:
        logger.info("[dry-run] Would write %s rows to %s", len(out_df), output_path)
    else:
        out_df.to_parquet(output_path, filesystem=config.fs)
        logger.info("Wrote %s rows to %s", len(out_df), output_path)

    return len(out_df)


def output_path_for_input(input_path: str, output_dir: str) -> str:
    """Build the output shard path from the input shard name."""
    stem = Path(input_path).stem
    match = re.search(r"(\d+)", stem)
    index = match.group(1).zfill(5) if match else stem
    return f"{output_dir.rstrip('/')}/shard_{index}.parquet"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build RAG v2 structured parts.")
    parser.add_argument("--input-dir", default=config.INPUT_DIR)
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--limit-shards", type=int, default=0)
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the preprocessing pipeline."""
    args = parse_args()

    input_shards = sorted([f for f in config.fs.ls(args.input_dir) if f.endswith(".parquet")])
    if args.limit_shards:
        input_shards = input_shards[: args.limit_shards]

    if not input_shards:
        logger.warning("No input shard found under %s", args.input_dir)
        return

    total_rows = 0
    for input_path in input_shards:
        output_path = output_path_for_input(input_path, args.output_dir)

        if not args.dry_run and not args.overwrite and config.fs.exists(output_path):
            logger.info("Output already exists, skipping: %s", output_path)
            continue

        total_rows += process_shard(
            input_path=input_path,
            output_path=output_path,
            limit_docs=args.limit_docs,
            dry_run=args.dry_run,
        )

    logger.info("Preprocessing finished. Total rows: %d", total_rows)


if __name__ == "__main__":
    main()
