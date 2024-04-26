from beyondllm.retrievers.base import BaseRetriever
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext

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
    def __init__(self, data, embed_model, top_k, vectordb,*args, **kwargs):
        """
        Initializes a NormalRetriever instance.

        Args:
            data: The dataset to be indexed.
            embed_model: The embedding model to use.
            top_k: The number of top results to retrieve.
            vectordb: The vectordb to use for retrieval
        """
        super().__init__(data, embed_model, vectordb,*args, **kwargs)
        self.embed_model = embed_model
        self.data = data
        self.top_k = top_k
        self.vectordb = vectordb

    def load_index(self):
        if self.data is None:
            index = self.initialize_from_vector_store()
        else:
            index = self.initialize_from_data()

        return index
    
    def initialize_from_vector_store(self):
        if self.vectordb is None:
            raise ValueError("Vector store must be provided if no data is passed")
        else:
            index = VectorStoreIndex.from_vector_store(
                self.vectordb,
                embed_model=self.embed_model,
            )
        return index


    def initialize_from_data(self):
        if self.vectordb==None:
            index = VectorStoreIndex(
                self.data, embed_model=self.embed_model
            )
        else:
            storage_context = StorageContext.from_defaults(vector_store=self.vectordb)
            index = VectorStoreIndex(
                self.data, storage_context=storage_context, embed_model=self.embed_model
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