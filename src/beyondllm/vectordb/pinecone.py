from beyondllm.vectordb.base import VectorDb, VectorDbConfig
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")
import os, subprocess, sys

@dataclass
class PineconeVectorDb:
    """
    from beyondllm.vectordb import PineconeVectorDb
    # to load pre-existing index
    vectordb = PineconeVectorDb(
                    api_key="<your api key>",
                    index_name="quickstart", 
                )
    # or 
    # create new index
    vectordb = PineconeVectorDb(create=True,
                    api_key="<your api key>", index_name="quickstart", 
                    embedding_dim=1536, metric="cosine",
                    spec="serverless", cloud="aws", region="us-east-1"
                )
    """

    api_key: str
    index_name: str
    create: bool = False
    embedding_dim: int = None
    metric: str = None
    spec: str = "serverless"
    cloud: str = None
    region: str = None
    pod_type: str = None  # Only required for pod-type Pinecone indexes
    replicas: int = None  # Only required for pod-type Pinecone indexes

    def __post_init__(self):
        try:
            from llama_index.vector_stores.pinecone import PineconeVectorStore
            from pinecone import Pinecone, ServerlessSpec
        except ImportError:
            user_agree = input(
                "The feature you're trying to use requires an additional library(s):llama_index.vector_stores.pinecone, pinecone-client. Would you like to install it now? [y/N]: "
            )
            if user_agree.lower() == "y":
                subprocess.check_call([sys.executable, "-m", "pip", "install", "llama_index.vector_stores.pinecone"])
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pinecone-client"])
                from llama_index.vector_stores.pinecone import PineconeVectorStore
                from pinecone import Pinecone, ServerlessSpec
            else:
                raise ImportError("The required 'llama_index.vector_stores.pinecone and pinecone-client' are not installed.")

        pinecone_client = Pinecone(api_key=self.api_key)

        if self.create==False:
            self.pinecone_index = pinecone_client.Index(self.index_name)
        else:
            if self.spec is "pod-based" or None:
                from pinecone import PodSpec

                pinecone_client.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dim,
                    metric=self.metric,
                    spec=PodSpec(environment=self.environment, pod_type=self.pod_type, replicas=self.replicas),
                )
            else:
                pinecone_client.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dim,
                    metric=self.metric,
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                )
                
            self.pinecone_index = pinecone_client.Index(self.index_name)

        self.load()

    def load(self):
        try:
            from llama_index.vector_stores.pinecone import PineconeVectorStore
        except:
            raise ImportError(
                "PineconeVectorStore library is not installed. Please install it with ``pip install llama_index.vector_stores.pinecone``."
            )

        try:
            vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
            self.client = vector_store
        except Exception as e:
            raise Exception(f"Failed to load the Pinecone Vectorstore: {e}")

        return self.client

    def add(self, *args, **kwargs):
        client = self.client
        return client.add(*args, **kwargs)

    def stores_text(self, *args, **kwargs):
        client = self.client
        return client.stores_text(*args, **kwargs)

    def is_embedding_query(self, *args, **kwargs):
        client = self.client
        return client.is_embedding_query(*args, **kwargs)

    def query(self, *args, **kwargs):
        client = self.client
        return client.query(*args, **kwargs)

    @staticmethod
    def load_from_kwargs(self, kwargs):
        embed_config = VectorDbConfig(**kwargs)
        self.config = embed_config
        self.load()