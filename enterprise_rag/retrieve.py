from .retrievers.normalRetriever import NormalRetriever
from .retrievers.crossEncoderReranker import CrossEncoderRerankRetriever

def auto_retriever(data,embed_model,type="normal",top_k=4,**kwargs):
    if type == 'normal':
        retriever = NormalRetriever(data,embed_model,top_k,**kwargs)
    elif type == 'flag-rerank':
        from .retrievers.flagReranker import FlagEmbeddingRerankRetriever
        retriever = FlagEmbeddingRerankRetriever(data,embed_model,top_k,**kwargs)
    elif type == 'cross-rerank':
        retriever = CrossEncoderRerankRetriever(data,embed_model,top_k,**kwargs)

    return retriever