import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from enterprise_rag.embeddings import FastEmbedEmbeddings
from .retrievers.normalRetriever import NormalRetriever

def auto_retriever(data,embed_model=None,type="normal",top_k=4,**kwargs):
    """
    Automatically selects and initializes a retriever based on the specified type.
    Parameters:
        data: The dataset to be used by the retriever for information retrieval.
        embed_model: The embedding model to be used for generating embeddings. 
           - Default Embedding: Fast Embeddings
        type (str): The type of retriever to use. Options include 'normal', 'flag-rerank', 
                    'cross-rerank', and 'hybrid'. Defaults to 'normal'.
        top_k (int): The number of top results to retrieve. Defaults to 4.
    Additional parameters:
        reranker: Name of the reranking model to be used. To be specified only for type = 'flag-rerank' and 'cross-rerank'
        mode: Possible options are 'AND' or 'OR'. To be specified only for type = 'hybrid. 'AND' mode will retrieve nodes in common between
                    keyword and vector retriever. top_k will be overridden in this case.
    Returns:
        A BaseRetriever instance.
    Example:
        from enterprise_rag.retrieve import auto_retriever
        
        data = <your dataset here>
        embed_model = <pass your embed model here>
        
        retriever = auto_retriever(data=data, embed_model=embed_model, type="normal", top_k=5)
    """
    if embed_model is None:
        embed_model = FastEmbedEmbeddings()
    if type == 'normal':
        retriever = NormalRetriever(data,embed_model,top_k,**kwargs)
    elif type == 'flag-rerank':
        from .retrievers.flagReranker import FlagEmbeddingRerankRetriever
        retriever = FlagEmbeddingRerankRetriever(data,embed_model,top_k,**kwargs)
    elif type == 'cross-rerank':
        from .retrievers.crossEncoderReranker import CrossEncoderRerankRetriever
        retriever = CrossEncoderRerankRetriever(data,embed_model,top_k,**kwargs)
    elif type == 'hybrid':
        from .retrievers.hybridRetriever import HybridRetriever
        retriever = HybridRetriever(data,embed_model,top_k,**kwargs)

    return retriever
