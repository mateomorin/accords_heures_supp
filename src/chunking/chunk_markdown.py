"""
Script originaly computed by Gemini to chunk data using langchain.
Reviewed and adapted for the project's purposes:
    - Parquet file reading to extract markdown and metadata
    - Filters by theme or by year
    - Embedding transformation using LLM Lab API
    - Changed commentaries for reproducibility
    - Linting
"""
import argparse
import asyncio
import logging
from tqdm import tqdm
import os
from typing import Any

import pandas as pd
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from qdrant_client import models

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("httpx2").setLevel(logging.WARNING)
logging.getLogger("docling").setLevel(logging.WARNING)


def parse_theme_filter(value):
    """
    Either all themes or just some to keep
    """
    if value.lower() == "all":
        return "all"

    try:
        themes = [str(m.strip('"')) for m in value.split('","')]
        return themes
    except ValueError:
        raise argparse.ArgumentTypeError(
            """Themes must be 'all' or themes delemited by '"' and split by commas."""
        )


def apply_default_collection(data: str, themes: str | list[str]):
    """
    Outputs data_theme1_theme_2..., or data_all
    """
    if isinstance(themes, list[str]):
        return data + "_".join(themes)
    else:
        return data


def create_qdrant_collection(collection_name: str):
    """
    Create the Qdrant collection associated with collection name if it does not exist.
    """
    if not config.qdrant_client.collection_exists(collection_name=collection_name):
        config.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=config.EMBEDDING_SIZE,
                distance=models.Distance.COSINE
            )
        )


def gather_shards(path: str) -> pd.DataFrame:
    """
    Fetch all data in the path and concatenate them into one dataframe.
    """
    shards = []
    for file in config.fs.ls(path):
        shards.append(pd.read_parquet("s3://" + file, filesystem=config.fs))
    return pd.concat(shards)


def clean_database(df: pd.DataFrame):
    """
    Remove unnecessary columns. Rename columns of interest
    """
    df.drop(
        columns=[
            "id",
            "legalStatus",
            "startDate",
            "endDate",
            "nature",
        ],
        inplace=True
    )

    df.rename(
        columns={
            "YYYY": "anneeSignature",
            "MM": "moisSignature",
        },
        inplace=True
    )
    return df


def filter_database_by_theme(df: pd.DataFrame, themes: list[str] | str):
    """
    Keep data only belonging to certain themes.
    """
    if isinstance(themes, str) and themes == "all":
        return df
    if isinstance(themes, str) and themes != "all":
        raise TypeError(f"themes {themes} is of type str but not equal to all")
    return df[df[themes].any(axis=1)]


def split_markdown_metadata(df: pd.DataFrame):
    markdowns = df["markdown"].to_list()
    metadata = df.drop(columns=["markdown"]).reset_index(drop=True).to_dict(orient="index").values()

    return markdowns, list(metadata)


def chunk_markdown_with_headers(
    markdown_content: str,
    extra_metadata: dict[str, Any],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP
) -> list[Document]:
    """
    Chunk markdown, preverving global structure when possible (header splitting).
    If parts are too big, split into paragraphs, then sentences, then lists, then words.
    """

    # Step A: split on Markdown headers
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    # Step B: split on text separators if paragraph is too long
    # Ordered separators:
    # 1. New paragraph/section
    # 2. End of sentence
    # 3. End of list item
    # 4. End of word
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            ".\n",
            ". ",
            "\n- ",
            "\n1. ",
            " ",
            ""
        ]
    )

    # Chunking

    final_documents: list[Document] = []
    header_splits = markdown_splitter.split_text(markdown_content)

    for doc in header_splits:
        sub_docs = recursive_splitter.split_documents([doc])
        for sub_doc in sub_docs:
            # Use all metadata
            combined_metadata = {**sub_doc.metadata, **extra_metadata}

            # Mention headers of the current paragraph
            breadcrumb = " > ".join([
                v for k, v in sub_doc.metadata.items() if k.startswith("Header_")
            ])

            content_with_context = sub_doc.page_content
            if breadcrumb:
                content_with_context = f"[Contexte: {breadcrumb}]\n{sub_doc.page_content}"

            final_documents.append(
                Document(
                    page_content=content_with_context,
                    metadata=combined_metadata
                )
            )

    return final_documents


async def embed_chunks(chunks: list[str], semaphore):
    """
    Embed chunks asynchronously using semaphore
    """
    async with semaphore:
        responses = await config.llm_client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=chunks
        )
        return [responses.data[i].embedding for i in range(len(chunks))]


async def embed_documents(documents: list):
    """
    Embed several documents at once using asyncio semaphore
    """
    semaphore = asyncio.Semaphore(config.EMBED_CONCURRENCY)
    tasks = [embed_chunks(chunks, semaphore) for chunks in documents]

    return await asyncio.gather(*tasks)


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Scrapping of ACCO by month and year"
    )

    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Name of the data floder containing all the shards (for a specific year)"
    )

    parser.add_argument(
        "--themes",
        type=parse_theme_filter,
        required=False,
        help="""Specific themes, quoted ("") and divided by coma, or 'all'""",
        default="all"
    )

    parser.add_argument(
        "--collection",
        type=str,
        required=False,
        help="Collection name to store vectors in Qdrant.",
        default="default"
    )

    args = parser.parse_args()

    if args.collection == "default":
        args.collection = apply_default_collection(data=args.data_folder, themes=args.themes)

    logger.info("Fetching data...")
    all_accords = gather_shards(config.FULL_DATA_PATH + args.data_folder)
    all_accords = clean_database(all_accords)
    all_accords = filter_database_by_theme(all_accords, args.themes)
    markdowns, metadata = split_markdown_metadata(df=all_accords)
    logger.info("Done!")

    logger.info("Converting database to markdown...")
    all_docs = []
    for doc_markdown, doc_metadata in tqdm(zip(markdowns, metadata)):
        docs = chunk_markdown_with_headers(
            markdown_content=doc_markdown,
            extra_metadata=doc_metadata
        )
        all_docs += docs
    logger.info("Done!")

    logger.info(f"Sending data to Qdrant collection {args.collection}...")
    vector_store = QdrantVectorStore.from_documents(
        documents=all_docs,
        embedding=config.embeddings,
        url=os.environ.get("QDRANT_API_URL", "http://qdrant:6333"),
        api_key=os.environ.get("QDRANT_API_KEY", None),
        collection_name=args.collection,
        force_recreate=False
    )

    logger.info("Everything is finished!")


if __name__ == "__main__":
    main()
