from beyondllm.embeddings.base import BaseEmbeddings,EmbeddingConfig
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")
import os

@dataclass
class GeminiEmbeddings:
    """
    from beyondllm.embeddings import GeminiEmbeddings
    embed_model = GeminiEmbeddings(model_name="models/embedding-001",api_key="AI***")
    """
    api_key: str = ""
    model_name:  str = field(default='models/embedding-001')

    def __post_init__(self):
        if not self.api_key:  
            self.api_key = os.getenv('GOOGLE_API_KEY') 
            if not self.api_key: 
                raise ValueError("GOOGLE_API_KEY is not provided and not found in environment variables.")
        self.load()

    def load(self):
        try:
            from llama_index.embeddings.gemini import GeminiEmbedding
        except:
            raise ImportError("Gemini Embeddings library is not installed. Please install it with ``pip install llama-index-embeddings-gemini``.")
        
        try:
            self.client = GeminiEmbedding(model_name=self.model_name,api_key = self.api_key)

        except Exception as e:
            raise Exception("Failed to load the embeddings from Gemini:", str(e))
        
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
