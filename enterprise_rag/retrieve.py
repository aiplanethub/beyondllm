from .retrievers.normalRetriever import NormalRetriever

def auto_retriever(data,embed_model,type="normal",top_k=4,**kwargs):
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