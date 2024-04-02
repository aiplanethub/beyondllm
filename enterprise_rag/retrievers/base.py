from .evaluate import evaluate_retriever

class BaseRetriever:
    def __init__(self, data, embed_model,**kwargs):
        self.data = data
        self.embedder = embed_model

    def load_index(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def as_retriever(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def retrieve(self,query):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def evaluate(self,llm):
        nodes = self.data
        retriever = self.as_retriever()
        return evaluate_retriever(llm,nodes,retriever)