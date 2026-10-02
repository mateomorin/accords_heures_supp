import os

import s3fs
from dotenv import load_dotenv
from openai import AsyncOpenAI
from qdrant_client import QdrantClient

load_dotenv(override=True)

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Data storage
fs = s3fs.S3FileSystem(
    endpoint_url="https://minio.lab.sspcloud.fr",
    client_kwargs={"region_name": "us-east-1"},
)
FULL_DATA_PATH = "s3://mateomorin/legifrance/data/"

# Embedding creation
EMBEDDING_MODEL = "qwen3-embedding-8b"
EMBEDDING_SIZE = 4096               # to be changed depending on the model
EMBED_CONCURRENCY = 10

llm_client = AsyncOpenAI(
    base_url=os.environ["LLM_API_URL"],
    api_key=os.environ["LLM_API_KEY"],
)


# Vector storage
qdrant_client = QdrantClient(
    url=os.environ.get("QDRANT_API_URL", "http://qdrant:6333"),
    api_key=os.environ.get("QDRANT_API_KEY", None),
    timeout=60
)
