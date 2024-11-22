from typing import Any
from beyondllm.vectordb.base import VectorDbConfig
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings("ignore")
import subprocess, sys


@dataclass
class QdrantVectorDb:
    """
    Vector store implementation using Qdrant - https://qdrant.tech/.
    REQUIRES the `"llama-index-vector-stores-qdrant"` package.

    >>> from beyondllm.vectordb import QdrantVectorDb
    >>> from qdrant_client import QdrantClient

    >>> vdb = QdrantVectorDb(
    ...     collection_name="my-collection-name",
    ...     client=QdrantClient(url="http://localhost:6333"),
    ...     llamaindex_kwargs={
    ...         "batch_size": 64
    ...     }
    ... )
    """

    client: Any
    collection_name: str
    llamaindex_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        try:
            from llama_index.vector_stores.qdrant import QdrantVectorStore

            self.client: QdrantVectorStore

        except ImportError:
            user_agree = input(
                "The feature requires an additional package: llama-index-vector-stores-qdrant. Would you like to install it now? [y/n]: "
            )
            if user_agree.lower() == "y":
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "pip install llama-index-vector-stores-qdrant",
                    ]
                )
                from llama_index.vector_stores.qdrant import QdrantVectorStore
            else:
                raise ImportError(
                    "The 'llama-index-vector-stores-qdrant' package couldn't be installed."
                )

        self.load()

    def load(self):
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        self.client = QdrantVectorStore(
            collection_name=self.collection_name,
            client=self.client,
            **self.llamaindex_kwargs,
        )
        return self.client

    def add(self, *args, **kwargs):
        return self.client.add(*args, **kwargs)

    def stores_text(self, *args, **kwargs):
        return self.client.stores_text(*args, **kwargs)

    def is_embedding_query(self, *args, **kwargs):
        return self.client.is_embedding_query(*args, **kwargs)

    def query(self, *args, **kwargs):
        return self.client.query(*args, **kwargs)

    @staticmethod
    def load_from_kwargs(self, kwargs):
        embed_config = VectorDbConfig(**kwargs)
        self.config = embed_config
        self.load()
