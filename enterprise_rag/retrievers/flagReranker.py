from .base import BaseRetriever
from llama_index.core import VectorStoreIndex, ServiceContext
# try:
#     from llama_index.core.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
# except:
#     raise ImportError("flag_embedding_reranker library is not installed. Please install it with ``pip install llama-index-postprocessor-flag-embedding-reranker`` and ``pip install FlagEmbedding``.")
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

from llama_index.core.schema import QueryBundle

class FlagEmbeddingRerankRetriever(BaseRetriever):
    def __init__(self, data, embed_model, top_k,*args, **kwargs):
        super().__init__(data, embed_model,*args, **kwargs)
        self.embed_model = embed_model
        self.data = data
        self.top_k = top_k

    def load_index(self):
        service_context = ServiceContext.from_defaults(llm=None, embed_model=self.embed_model)
        index = VectorStoreIndex(
            self.data, service_context= service_context,
        )
        return index
    
    def retrieve(self, query):
        retriever = self.as_retriever()
        unranked_nodes = retriever.retrieve(query)

        reranker = FlagEmbeddingReranker(
            top_n=self.top_k,
            model="BAAI/bge-reranker-large",
        )

        query_bundle = QueryBundle(query_str=query)
        ranked_nodes = reranker._postprocess_nodes(unranked_nodes, query_bundle = query_bundle)

        return ranked_nodes

    def as_retriever(self):
        index = self.load_index()

        reranker = FlagEmbeddingReranker(
            top_n=self.top_k,
            model="BAAI/bge-reranker-base",
        )

        retriever_index= index.as_retriever(
            similarity_top_k=self.top_k,
            node_postprocessors=[reranker]
            )
        return retriever_index

    def as_query_engine(self):
        index = self.load_index()

        query_engine = index.as_query_engine()
        return query_engine

    def query(self, user_query):
        query_engine = self.as_query_engine()

        return query_engine(user_query)
    