from beyondllm.retrievers.base import BaseRetriever
from llama_index.core import VectorStoreIndex, ServiceContext

class NormalRetriever(BaseRetriever):
    """
    A simple retriever that uses vector similarity search to retrieve relevant documents.

    Example:
        from beyondllm.retrieve import auto_retrievers

        data = ...  # Load your data
        embed_model = # Load your embed model
        retriever = auto_retriever(data=data, embed_model=embed_model, type="normal", top_k=5)

        results = retriever.retrieve("<your query>")
    """
    def __init__(self, data, embed_model, top_k,*args, **kwargs):
        """
        Initializes a NormalRetriever instance.

        Args:
            data: The dataset to be indexed.
            embed_model: The embedding model to use.
            top_k: The number of top results to retrieve.
        """
        super().__init__(data, embed_model,*args, **kwargs)
        self.embed_model = embed_model
        self.data = data
        self.top_k = top_k

    def load_index(self):
        service_context = ServiceContext.from_defaults(llm=None, embed_model=self.embed_model)
        index = VectorStoreIndex(
            self.data, service_context= service_context
        )
        return index
    
    def retrieve(self, query):
        retriever = self.as_retriever()
        return retriever.retrieve(query)

    def as_retriever(self):
        index = self.load_index()

        return index.as_retriever(similarity_top_k=self.top_k)

    def as_query_engine(self):
        index = self.load_index()

        query_engine = index.as_query_engine()
        return query_engine

    def query(self, user_query):
        query_engine = self.as_query_engine()

        return query_engine(user_query)