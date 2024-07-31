from beyondllm.vectordb.base import VectorDb, VectorDbConfig
from dataclasses import dataclass
import warnings
import subprocess
import sys

warnings.filterwarnings("ignore")

@dataclass
class WeaviateVectorDb:
    """
    from beyondllm.vectordb import WeaviateVectorDb
    vectordb = WeaviateVectorDb(
        url="<cluster_url>",
        index_name="MyClass",
        api_key="<your api key>",
        additional_headers=None
    )
    """
    url: str
    index_name: str
    api_key: str = None  # Optional, for authenticated access
    additional_headers: dict = None  # Optional, for custom headers

    def __post_init__(self):
        try:
            from llama_index.vector_stores.weaviate import WeaviateVectorStore
            import weaviate
        except ImportError:
            user_agree = input("The feature you're trying to use requires additional libraries: llama_index.vector_stores.weaviate and weaviate-client. Would you like to install them now? [y/N]: ")
            if user_agree.lower() == 'y':
                subprocess.check_call([sys.executable, "-m", "pip", "install", "llama_index.vector_stores.weaviate"])
                subprocess.check_call([sys.executable, "-m", "pip", "install", "weaviate-client"])
                from llama_index.vector_stores.weaviate import WeaviateVectorStore
                import weaviate
            else:
                raise ImportError("The required 'llama_index.vector_stores.weaviate' and 'weaviate-client' are not installed.")

        auth_config = weaviate.auth.AuthApiKey(api_key=self.api_key) if self.api_key else None
        self.client = weaviate.connect_to_wcs(
            cluster_url=self.url,
            auth_credentials=auth_config,
            headers=self.additional_headers
        )
        self.load()

    def load(self):
        try:
            from llama_index.vector_stores.weaviate import WeaviateVectorStore
        except ImportError:
            raise ImportError("WeaviateVectorStore library is not installed. Please install it with `pip install llama_index.vector_stores.weaviate`.")

        try:
            if not self.client.collections.exists(self.index_name):
                self.client.collections.create(
                    self.index_name,
                )

            vector_store = WeaviateVectorStore(
                weaviate_client=self.client,
                index_name=self.index_name,
            )
            self.client = vector_store
        except Exception as e:
            raise Exception(f"Failed to load the Weaviate Vectorstore: {e}")

        return self.client

    def add(self, *args, **kwargs):
        return self.client.add(*args, **kwargs)

    def stores_text(self, *args, **kwargs):
        return self.client.stores_text(*args, **kwargs)

    def is_embedding_query(self, *args, **kwargs):
        return self.client.is_embedding_query(*args, **kwargs)

    def query(self, *args, **kwargs):
        return self.client.query(*args, **kwargs)
    
    def insert_document(self, document):
        self.client.insert(document)

    def delete_index(self):
        self.client.delete_index()

    def close_connection(self):
        self.client.close()

    def load_from_kwargs(self, kwargs):
        embed_config = VectorDbConfig(**kwargs)
        self.config = embed_config
        self.load()