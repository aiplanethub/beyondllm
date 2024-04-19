from beyondllm.embeddings.base import BaseEmbeddings,EmbeddingConfig
import os
from dataclasses import dataclass, field

@dataclass
class HuggingFaceInferenceAPIEmbeddings:
    """
    from beyondllm.embeddings import HuggingFaceInferenceAPIEmbedding
    embed_model = HuggingFaceInferenceAPIEmbeddings(model_name="thenlper/gte-large")
    """
    api_key: str = ""
    model_name:  str = field(default='sentence-transformers/all-MiniLM-L6-v2')

    def __post_init__(self):
        if not self.api_key:  
            self.api_key = os.getenv('HUGGINGFACE_ACCESS_TOKEN') 
            if not self.api_key: 
                raise ValueError("HUGGINGFACE_ACCESS_TOKEN is not provided and not found in environment variables.")
        self.load()

    def load(self):
        try:
            from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
        except:
            raise ImportError("HuggingFace Embeddings library is not installed. Please install it with ``pip install llama_index.embeddings.huggingface ``.")

        try:
            self.client = HuggingFaceInferenceAPIEmbedding(model_name=self.model_name,api_key=self.api_key)

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
    def load_from_kwargs(self,kwargs): 
        embed_config = EmbeddingConfig(**kwargs)
        self.config = embed_config
        self.load()