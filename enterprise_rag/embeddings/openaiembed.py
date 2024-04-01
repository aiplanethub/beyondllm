from .base import BaseEmbeddings,EmbeddingConfig
from pydantic import Field
from typing import Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

@dataclass
class OpenAIEmbeddings:
    """
    from enterprise_rag.embeddings import OpenAIEmbeddings
    embed_model = OpenAIEmbeddings(model_name="text-embedding-3-small",api_key="sk-")
    """
    api_key: str
    model_name:  str = Field(default='text-embedding-3-small')

    def __post_init__(self):
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

    @staticmethod
    def load_from_kwargs(kwargs): 
        embed_config = EmbeddingConfig(**kwargs)
        self.config = embed_config
        self.load()