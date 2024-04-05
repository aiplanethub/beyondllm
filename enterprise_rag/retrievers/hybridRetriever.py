from .base import BaseRetriever
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext, SimpleKeywordTableIndex
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever as LlamaBaseRetriever
from llama_index.core.retrievers import VectorIndexRetriever, KeywordTableSimpleRetriever

from typing import List

from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine


class CustomRetriever(LlamaBaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "AND",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


class HybridRetriever(BaseRetriever):
    """
    A hybrid retriever that combines the results of vector similarity search and Keyword search
    to retrieve relevant documents.

    Example:
        from enterprise_rag.retrieve import auto_retriever

        data = ...  # Load your data
        embed_model = # Load your embed model
        retriever = auto_retriever(data=data, embed_model=embed_model, type="hybrid", top_k=5, mode="AND")

        results = retriever.retrieve("<your query>")
    """
    def __init__(self, data, embed_model, top_k,*args, **kwargs):
        """
        Initializes a HybridRetriever instance.

        Args:
            data: The dataset to be indexed.
            embed_model: The embedding model to use.
            top_k: The number of top results to retrieve.
            mode: Possible options are 'AND' or 'OR'. To be specified only for type = 'hybrid. 'AND' mode will retrieve nodes in common between
                    keyword and vector retriever. top_k will be overridden in this case.
        """
        super().__init__(data, embed_model,*args, **kwargs)
        self.embed_model = embed_model
        self.data = data
        self.top_k = top_k
        self.mode = kwargs.get('mode', 'AND')  

        if self.mode not in ['AND', 'OR']:
            raise ValueError("Invalid mode. Mode must be 'AND' or 'OR'.")

    def load_index(self):
        service_context = ServiceContext.from_defaults(llm=None, embed_model=self.embed_model)
        storage_context = StorageContext.from_defaults()
        vector_index = VectorStoreIndex(
            self.data, service_context= service_context, storage_context=storage_context
        )
        keyword_index = SimpleKeywordTableIndex(
            self.data,service_context=service_context,storage_context=storage_context
        )
        return vector_index, keyword_index
    
    def as_retriever(self):
        vector_index, keyword_index = self.load_index()

        vector_retriever = vector_index.as_retriever(similarity_top_k=self.top_k)
        keyword_retriever = keyword_index.as_retriever(similarity_top_k=self.top_k)

        custom_retriever = CustomRetriever(vector_retriever, keyword_retriever, self.mode)

        return custom_retriever
    
    def retrieve(self, query):
        """
        Method to retrieve relevant nodes for a user's query using vector search.
        
        Example:
        from enterprise_rag.retrieve import auto_retrievers

        data = ...  # Load your data
        embed_model = # Load your embed model
        retriever = auto_retriever(data=data, embed_model=embed_model, type="normal", top_k=5)

        relevant_nodes = retriever.retrieve("<your query>")
        """
        custom_retriever = self.as_retriever()

        retrieve_nodes = custom_retriever.retrieve(query)

        return retrieve_nodes

    def as_query_engine(self,mode="AND"):
        vector_retriever, keyword_retriever = self.as_retriever()

        custom_retriever = CustomRetriever(vector_retriever, keyword_retriever, mode)

        response_synthesizer = get_response_synthesizer()

        custom_query_engine = RetrieverQueryEngine(
            retriever=custom_retriever,
            response_synthesizer=response_synthesizer,
        )
        return custom_query_engine

    def query(self, user_query):
        custom_query_engine = self.as_query_engine()

        return custom_query_engine(user_query)
    