from pydantic import BaseModel

class EmbeddingConfig(BaseModel):
    """Base configuration model for all LLMs.

    This class can be extended to include more fields specific to certain LLMs.
    """
    pass 

class BaseEmbeddings(BaseModel):
    def __init__(self, model_name="AzureOpenAI",**kwargs):
        self.model_name = model_name

    def load(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def embed_text(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def embed_text(self,text):
        embed_model = self.load()
        return embed_model.embed_text(text)

    def get_text_embedding_batch(self, *args, **kwargs):
        """
        Get embeddings in batch from Azure OpenAI.
        """
        embed_model = self.load()
        return embed_model.get_text_embedding_batch(*args, **kwargs)
    
    def get_agg_embedding_from_queries(self,*args,**kwargs):
        embed_model = self.load()
        return embed_model.get_agg_embedding_from_queries(*args,**kwargs)
