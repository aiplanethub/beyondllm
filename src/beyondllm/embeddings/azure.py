from beyondllm.embeddings.base import EmbeddingConfig
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

@dataclass
class AzureAIEmbeddings:
    """
    from beyondllm.embeddings import AzureAIEmbeddings
    embed_model = AzureAIEmbeddings(endpoint_url="https:---",azure_key="",api_version="",deployment_name="")
    """
    azure_key: str
    endpoint_url: str
    api_version: str
    deployment_name: str

    def __post_init__(self):
        self.load()

    def load(self):
        try:
            from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
        except:
            raise ImportError("Azure OpenAI Embeddings library is not installed. Please install it with ``pip install llama-index-embeddings-azure_openai``.")
        
        try:
            self.client = AzureOpenAIEmbedding(
                api_key=self.azure_key,
                azure_endpoint = self.endpoint_url,
                api_version = self.api_version,
                deployment_name = self.deployment_name
            )

        except Exception as e:
            raise Exception("Failed to load the embeddings from AzureOpenAI:", str(e))
        
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