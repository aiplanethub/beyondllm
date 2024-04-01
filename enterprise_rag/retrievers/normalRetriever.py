from .base import BaseRetriever
from llama_index.core import VectorStoreIndex, ServiceContext


class NormalRetriever(BaseRetriever):
    def __init__(self, data, embed_model, top_k,*args, **kwargs):
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
    