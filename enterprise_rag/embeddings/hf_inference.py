from .base import BaseEmbeddings,EmbeddingConfig
from typing import Any, Optional
from dataclasses import dataclass, field

@dataclass
class HuggingFaceInferenceAPIEmbeddings:
    """
    from enterprise_rag.embeddings import HuggingFaceInferenceAPIEmbedding
    embed_model = HuggingFaceInferenceAPIEmbeddings(model_name="")
    """
    model_name:  str = field(default='BAAI/bge-small-en-v1.5')

    def __post_init__(self):
        self.load()

    def load(self):
        try:
            from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
        except:
            raise ImportError("HuggingFace Embeddings library is not installed. Please install it with ``pip install llama_index.embeddings.huggingface ``.")
        
        try:
            self.client = HuggingFaceInferenceAPIEmbedding(model_name=self.model_name)

        except Exception as e:
            raise Exception("Failed to load the embeddings from HuggingFace:", str(e))
        
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
    def load_from_kwargs(kwargs): 
        embed_config = EmbeddingConfig(**kwargs)
        self.config = embed_config
        self.load()