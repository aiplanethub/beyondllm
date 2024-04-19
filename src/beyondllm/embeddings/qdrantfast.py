from beyondllm.embeddings.base import BaseEmbeddings,EmbeddingConfig
from dataclasses import dataclass,field
import warnings
warnings.filterwarnings("ignore")

@dataclass
class FastEmbedEmbeddings:
    """
    from beyondllm.embeddings import FastEmbedEmbeddings
    embed_model = FastEmbedEmbeddings()
    """
    model_name:  str = field(default='BAAI/bge-small-en-v1.5')

    def __post_init__(self):
        self.load()

    def load(self):
        try:
            from llama_index.embeddings.fastembed import FastEmbedEmbedding
        except:
            raise ImportError("Qdrant Fast Embeddings Embeddings library is not installed. Please install it with ``pip install llama-index-embeddings-fastembed``.")
        
        try:
            self.client = FastEmbedEmbedding(model_name=self.model_name)

        except Exception as e:
            raise Exception("Failed to load the embeddings from Fast Embeddings:", str(e))
        
        return self.client
    
    def embed_text(self,text):
        embeds = self.client.get_text_embedding(text) 
        return embeds
    
    def get_text_embedding_batch(self,*args, **kwargs):
        batch = self.client.get_text_embedding_batch(*args, **kwargs)
        return batch
    
    def get_agg_embedding_from_queries(self,*args,**kwargs):
        agg_embedding = self.client.get_agg_embedding_from_queries(*args, **kwargs)
        return agg_embedding

    @staticmethod
    def load_from_kwargs(self,kwargs): 
        embed_config = EmbeddingConfig(**kwargs)
        self.config = embed_config
        self.load()