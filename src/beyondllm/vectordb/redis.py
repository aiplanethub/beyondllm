from beyondllm.vectordb.base import VectorDb, VectorDbConfig
from dataclasses import dataclass
import warnings
import subprocess
import sys

warnings.filterwarnings("ignore")

@dataclass
class RedisVectorDb:
    """
    Example usage:
    from beyondllm.vectordb import RedisVectorDb
    vectordb = RedisVectorDb(
        host="localhost",
        port=6379,
        index_name="MyIndex",
        password=None,
        schema=None,
        additional_headers=None
    )
    """
    host: str
    port: int
    index_name: str
    password: str = None  # Optional, for authenticated access
    schema: dict = None  # Optional, for custom schema
    additional_headers: dict = None  # Optional, for custom headers

    def __post_init__(self):
        try:
            from llama_index.vector_stores.redis import RedisVectorStore
            from redisvl.schema import IndexSchema
            import redis
        except ImportError:
            user_agree = input("The feature you're trying to use requires additional libraries: llama_index.vector_stores.redis and redis. Would you like to install them now? [y/N]: ")
            if user_agree.lower() == 'y':
                subprocess.check_call([sys.executable, "-m", "pip", "install", "llama_index.vector_stores.redis", "redis"])
                from llama_index.vector_stores.redis import RedisVectorStore
                from redisvl.schema import IndexSchema
                import redis
            else:
                raise ImportError("The required 'llama_index.vector_stores.redis' and 'redis' libraries are not installed.")

        self.client = redis.Redis(
            host=self.host,
            port=self.port,
            password=self.password,
            decode_responses=True
        )
        self.load()

    def load(self):
        try:
            from llama_index.vector_stores.redis import RedisVectorStore
            from redisvl.schema import IndexSchema
        except ImportError:
            raise ImportError("RedisVectorStore library is not installed. Please install it with `pip install llama_index.vector_stores.redis`.")

        try:
            if self.schema is None:
                self.schema = {
                    "index": {
                        "name": self.index_name,
                        "prefix": "essay",
                        "key_separator": ":",
                    },
                    "fields": [
                        {"type": "tag", "name": "id"},
                        {"type": "tag", "name": "doc_id"},
                        {"type": "text", "name": "text"},
                        {"type": "numeric", "name": "updated_at"},
                        {"type": "tag", "name": "file_name"},
                        {
                            "type": "vector",
                            "name": "vector",
                            "attrs": {
                                "dims": 768,
                                "algorithm": "hnsw",
                                "distance_metric": "cosine",
                            },
                        },
                    ],
                }

            index_schema = IndexSchema.from_dict(self.schema)
            vector_store = RedisVectorStore(
                redis_client=self.client,
                schema=index_schema,
                overwrite=True
            )
            self.client = vector_store
        except Exception as e:
            raise Exception(f"Failed to load the Redis Vectorstore: {e}")

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