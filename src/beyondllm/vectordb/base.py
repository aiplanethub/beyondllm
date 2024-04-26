from pydantic import BaseModel

class VectorDbConfig(BaseModel):
    """Base configuration model for all LLMs.

    This class can be extended to include more fields specific to certain LLMs.
    """
    pass 

class VectorDb(BaseModel):
    def load(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def add(self,*args, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def stores_text(self,*args, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def is_embedding_query(self,*args, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def query(self,*args, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")