"""
Shared clients and constants for the population_concernee visualisation pipeline.

Mirrors src/rag/config.py (same LLM/S3/Qdrant clients) and src/extraction/services.py,
adding the constants needed to embed the `population_concernee` column of the
contingent_evolution extraction into its own Qdrant collection.
"""
import os
from pathlib import Path

import s3fs
from dotenv import load_dotenv
from openai import AsyncOpenAI
from qdrant_client import QdrantClient

load_dotenv(override=True)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# Noisy-data curation list (anonymised references), as used by the notebook.
NOISY_PATH = REPO_ROOT / "notebooks/sources/noisy_data/accords_caches.txt"

# S3 base for the extraction outputs
EXTRACTION_BASE = "s3://mateomorin/legifrance/extraction"
METADATA_BASE = "s3://mateomorin/legifrance/metadata"

# Where the Qdrant collection lives / UMAP results are exported (overridable via CLI)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "notebooks/sources/results"

# --------------------------------------------------------------------------- #
# S3 / MinIO
# --------------------------------------------------------------------------- #
fs = s3fs.S3FileSystem(
    endpoint_url="https://minio.lab.sspcloud.fr",
    client_kwargs={"region_name": "us-east-1"},
)

# --------------------------------------------------------------------------- #
# Embedding creation (same model as the RAG pipeline)
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL = "qwen3-embedding-8b"
EMBEDDING_SIZE = 4096
EMBED_CONCURRENCY = 10
EMBED_BATCH_SIZE = 64

llm_client = AsyncOpenAI(
    base_url=os.environ["LLM_API_URL"],
    api_key=os.environ["LLM_API_KEY"],
)

# --------------------------------------------------------------------------- #
# Vector storage
# --------------------------------------------------------------------------- #
COLLECTION = "population_concernee_contingent"

qdrant_client = QdrantClient(
    url=os.environ.get("QDRANT_API_URL", "http://qdrant:6333"),
    api_key=os.environ.get("QDRANT_API_KEY", None),
    timeout=120,
)
