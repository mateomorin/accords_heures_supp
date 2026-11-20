"""Settings for the v2 parts ingestion (structure -> chunks -> Qdrant)."""

import os

import s3fs
from dotenv import load_dotenv
from openai import AsyncOpenAI
from qdrant_client import QdrantClient

load_dotenv(override=True)

# Input: one row per document, `parts_json` holds the structural parts (see src/preprocessing).
PARTS_PATH = "s3://mateomorin/legifrance/rag/v2/shards"

# Parts are embedded whole unless they exceed CHUNK_SIZE (median part = 455 chars).
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
MIN_CHUNK_CHARS = 30

# Vector storage
DEFAULT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "accords_heures_supp_parts")
EMBEDDING_MODEL = "qwen3-embedding-8b"
EMBEDDING_SIZE = 4096
EMBED_BATCH_SIZE = 64
EMBED_CONCURRENCY = 10
MAX_PARALLEL_UPSERTS = 4

# One `.done` marker per shard, written next to the input data.
CHECKPOINT_PATH = "s3://mateomorin/legifrance/rag/v2/checkpoints"

# Data storage
fs = s3fs.S3FileSystem(
    endpoint_url="https://minio.lab.sspcloud.fr",
    client_kwargs={"region_name": "us-east-1"},
)

# Embeddings
llm_client = AsyncOpenAI(
    base_url=os.environ["LLM_API_URL"],
    api_key=os.environ["LLM_API_KEY"],
)

# Vector storage
qdrant_client = QdrantClient(
    url=os.environ.get("QDRANT_API_URL", "http://qdrant:6333"),
    api_key=os.environ.get("QDRANT_API_KEY", None),
    timeout=60,
)
