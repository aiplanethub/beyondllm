from beyondllm.retrievers.utils import evaluate_retriever

class BaseRetriever:
    """
    A base class for creating retrievers that can load indexes and perform retrieval operations.
    
    Attributes:
        data: The dataset to be indexed or retrieved from.
        embed_model: The embedding model used to generate embeddings for the data.
        top_k: The top k similarity search results to be retrieved
        vectordb: The vectordb to be used for retrieval
    """
    def __init__(self, data, embed_model, vectordb, **kwargs):
        self.data = data
        self.embed_model = embed_model
        self.vectordb = vectordb

    def load_index(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def as_retriever(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def retrieve(self,query):
        """
        Method to retrieve relevant nodes for a user's query using vector search.
        
        Example:
        from beyondllm.retrieve import auto_retrievers

        data = ...  # Load your data
        embed_model = # Load your embed model
        retriever = auto_retriever(data=data, embed_model=embed_model, type="normal", top_k=5)

        relevant_nodes = retriever.retrieve("<your query>")
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def evaluate(self,llm):
        """
        Evaluates the performance of the retriever. Returns hit rate and MRR
        
        Example:
        from beyondllm.retrieve import auto_retriever
        
        data = # Load data using fit
        embed_model # Load your embedding model

        retriever = auto_retriever(data,embed_model,type="normal")
        results = retriever.evaluate(llm)

        Parameters:
            llm: The language model to use for QA dataset generation
            
        Returns:
            Dict: Contains Hit rate and MRR
        """
        nodes = self.data
        retriever = self.as_retriever()
        return evaluate_retriever(llm,nodes,retriever)