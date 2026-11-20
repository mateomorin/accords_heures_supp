"""Configuration for the RAG v2 preprocessing pipeline."""

import s3fs
from dotenv import load_dotenv

load_dotenv(override=True)

# S3 filesystem (MinIO)
fs = s3fs.S3FileSystem(
    endpoint_url="https://minio.lab.sspcloud.fr",
    client_kwargs={"region_name": "us-east-1"},
)

# Input / output locations
INPUT_DIR = "s3://mateomorin/legifrance/data_heures_supp/data/final_shards"
OUTPUT_DIR = "s3://mateomorin/legifrance/rag/v2/shards"
