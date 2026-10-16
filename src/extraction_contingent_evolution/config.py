import os
import s3fs

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv(override=True)


# S3 storage
fs = s3fs.S3FileSystem(
    endpoint_url="https://minio.lab.sspcloud.fr",
    client_kwargs={"region_name": "us-east-1"},
)
S3_OUTPUT_PATH = "s3://mateomorin/legifrance/extraction_contingent/"

# Embeddings
EMBEDDING_MODEL = "qwen3-embedding-8b"
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=os.environ["LLM_API_URL"],
    api_key=os.environ["LLM_API_KEY"],
    check_embedding_ctx_length=False
)

# Qdrant
QDRANT_COLLECTION = "accords_heures_supp"
qdrant_client = QdrantClient(
    url=os.environ.get("QDRANT_API_URL", "http://qdrant:6333"),
    api_key=os.environ.get("QDRANT_API_KEY", None),
    timeout=60
)

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=QDRANT_COLLECTION,
    embedding=embeddings,
)

YEAR_KEY = "metadata.anneeSignature"
REF_KEY = "metadata.reference"
DEFAULT_YEARS = list(range(2017, 2026))

# LLM
LLM_MODEL = "gemma4-26b-moe"
llm = ChatOpenAI(
    base_url=os.environ["LLM_API_URL"],
    api_key=os.environ["LLM_API_KEY"],
    model=LLM_MODEL,
    temperature=0,
    timeout=120
)
