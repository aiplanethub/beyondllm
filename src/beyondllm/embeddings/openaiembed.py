from beyondllm.embeddings.base import BaseEmbeddings,EmbeddingConfig
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")
import os

@dataclass
class OpenAIEmbeddings:
    """
    from beyondllm.embeddings import OpenAIEmbeddings
    embed_model = OpenAIEmbeddings(model_name="text-embedding-3-small",api_key="sk-")
    """
    api_key: str = ""
    model_name:  str = field(default='text-embedding-3-small')

    def __post_init__(self):
        if not self.api_key:  
            self.api_key = os.getenv('OPENAI_API_KEY') 
            if not self.api_key: 
                raise ValueError("OPENAI_API_KEY is not provided and not found in environment variables.")
        self.load()

    def load(self):
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
        except:
            raise ImportError("OpenAI Embeddings library is not installed. Please install it with ``pip install llama-index-embeddings-openai``.")
        
        try:
            self.client = OpenAIEmbedding(model_name=self.model_name,api_key = self.api_key)

        except Exception as e:
            raise Exception("Failed to load the embeddings from OpenAI:", str(e))
        
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
