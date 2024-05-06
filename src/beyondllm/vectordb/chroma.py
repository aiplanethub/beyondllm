from beyondllm.vectordb.base import VectorDb, VectorDbConfig
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")
import subprocess,sys

@dataclass
class ChromaVectorDb:
    """
    from beyondllm.vectordb import ChromaVectorDb
    vectordb = ChromaVectorDb(collection_name="quickstart",persist_directory="./db/chroma/")
    """
    collection_name: str
    persist_directory: str = ""

    def __post_init__(self):
        try:
            from llama_index.vector_stores.chroma import ChromaVectorStore
        except ImportError:
            user_agree = input("The feature you're trying to use requires an additional library(s):llama_index.vector_stores.chroma. Would you like to install it now? [y/N]: ")
            if user_agree.lower() == 'y':
                subprocess.check_call([sys.executable, "-m", "pip", "install", "llama_index.vector_stores.chroma"])
                from llama_index.vector_stores.chroma import ChromaVectorStore
            else:
                raise ImportError("The required 'llama_index.vector_stores.chroma' is not installed.")
        import chromadb
        
        if self.persist_directory=="" or self.persist_directory==None:  
            self.chroma_client = chromadb.EphemeralClient()
        else:
            self.chroma_client = chromadb.PersistentClient(self.persist_directory)
        self.load()

    def load(self):
        try:
            from llama_index.vector_stores.chroma import ChromaVectorStore
        except:
            raise ImportError("ChromaVectorStore library is not installed. Please install it with ``pip install llama_index.vector_stores.chroma``.")
        
        # More clarity and specificity required for try error statements
        try:
            try:
                chroma_collection = self.chroma_client.get_collection(self.collection_name)
            except Exception:
                chroma_collection = self.chroma_client.create_collection(self.collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            self.client = vector_store
        except Exception as e:
            raise Exception(f"Failed to load the Chroma Vectorstore: {e}")
        
        return self.client
    
    def add(self,*args, **kwargs):
        client = self.client
        return client.add(*args, **kwargs)
    
    def stores_text(self,*args, **kwargs):
        client = self.client
        return client.stores_text(*args, **kwargs)
    
    def is_embedding_query(self,*args, **kwargs):
        client = self.client
        return client.is_embedding_query(*args, **kwargs)
    
    def query(self,*args, **kwargs):
        client = self.client
        return client.query(*args, **kwargs)
    

    @staticmethod
    def load_from_kwargs(self,kwargs): 
        embed_config = VectorDbConfig(**kwargs)
        self.config = embed_config
        self.load()
