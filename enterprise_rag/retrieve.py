from .retrievers.normalRetriever import NormalRetriever

def auto_retriever(data,embed_model,type="normal",top_k=4,**kwargs):
    if type == 'normal':
        retriever = NormalRetriever(data,embed_model,top_k,**kwargs)

    return retriever