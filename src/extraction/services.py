import os

import s3fs
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from omegaconf import DictConfig
from qdrant_client import QdrantClient

load_dotenv(override=True)


class ServiceContainer:
    """
    Reusable container for all clients and external services.
    """
    def __init__(self, cfg: DictConfig):

        # S3 / MinIO
        self.fs = s3fs.S3FileSystem(
            endpoint_url="https://minio.lab.sspcloud.fr",
            client_kwargs={"region_name": "us-east-1"},
        )

        # Embeddings
        self.embeddings = OpenAIEmbeddings(
            model=cfg.llm.embedding_model,
            base_url=os.environ["LLM_API_URL"],
            api_key=os.environ["LLM_API_KEY"],
            check_embedding_ctx_length=False
        )

        # Vector storage
        self.qdrant_client = QdrantClient(
            url=os.environ.get("QDRANT_API_URL", "http://qdrant:6333"),
            api_key=os.environ.get("QDRANT_API_KEY", None),
            timeout=cfg.vector_storage.timeout
        )

        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=cfg.vector_storage.qdrant_collection,
            embedding=self.embeddings,
        )

        # LLM
        self.llm = ChatOpenAI(
            base_url=os.environ["LLM_API_URL"],
            api_key=os.environ["LLM_API_KEY"],
            model=cfg.llm.llm_model,
            temperature=cfg.llm.temperature,
            timeout=cfg.llm.timeout
        )
