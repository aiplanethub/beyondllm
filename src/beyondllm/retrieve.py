import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from beyondllm.embeddings import GeminiEmbeddings
from beyondllm.retrievers.normalRetriever import NormalRetriever
from beyondllm.retrievers.utils import generate_qa_dataset, evaluate_from_dataset
import pandas as pd

def auto_retriever(data=None,embed_model=None,type="normal",top_k=4,vectordb=None,**kwargs):
    """
    Automatically selects and initializes a retriever based on the specified type.
    Parameters:
        data: The dataset to be used by the retriever for information retrieval.
        embed_model: The embedding model to be used for generating embeddings. 
           - Default Embedding: Fast Embeddings
        type (str): The type of retriever to use. Options include 'normal', 'flag-rerank', 
                    'cross-rerank', and 'hybrid'. Defaults to 'normal'.
        top_k (int): The number of top results to retrieve. Defaults to 4.
        vectordb (VectorDb): The vectordb to use for retrieval
    Additional parameters:
        reranker: Name of the reranking model to be used. To be specified only for type = 'flag-rerank' and 'cross-rerank'
        mode: Possible options are 'AND' or 'OR'. To be specified only for type = 'hybrid. 'AND' mode will retrieve nodes in common between
                    keyword and vector retriever. top_k will be overridden in this case.
    Returns:
        A BaseRetriever instance.
    Example:
        from beyondllm.retrieve import auto_retriever
        
        data = <your dataset here>
        embed_model = <pass your embed model here>
        vector_store = <pass your vector-store object here>
        
        retriever = auto_retriever(data=data, embed_model=embed_model, type="normal", top_k=5, vectordb=vector_store)
    """
    if embed_model is None:
        embed_model = GeminiEmbeddings()
    if type == 'normal':
        retriever = NormalRetriever(data,embed_model,top_k,vectordb,**kwargs)
    elif type == 'flag-rerank':
        from .retrievers.flagReranker import FlagEmbeddingRerankRetriever
        retriever = FlagEmbeddingRerankRetriever(data,embed_model,top_k,vectordb,**kwargs)
    elif type == 'cross-rerank':
        from .retrievers.crossEncoderReranker import CrossEncoderRerankRetriever
        retriever = CrossEncoderRerankRetriever(data,embed_model,top_k,vectordb,**kwargs)
    elif type == 'hybrid':
        from .retrievers.hybridRetriever import HybridRetriever
        retriever = HybridRetriever(data,embed_model,top_k,vectordb,**kwargs)
    else:
        raise NotImplementedError(f"Retriever for the type '{type}' is not implemented.")

    return retriever

def compare_retrievers(data,llm, retrievers_list):
    """
    Compares multiple retrievers on a QA dataset, evaluating their hit rate and MRR.

    Parameters:
        data (List[TextNode]): The list of TextNodes to generate the QA dataset from.
        llm (BaseLLMModel): The language model used to generate the QA dataset.
        retrievers_list (List[BaseRetriever]): A list of retriever instances to evaluate.

    Returns:
        pandas.DataFrame: A DataFrame containing the hit rate and MRR for each retriever.
    """
    qa_dataset = generate_qa_dataset(llm, data)
    
    results = []

    for retriever in retrievers_list:
        hit_rate, mrr = evaluate_from_dataset(qa_dataset, retriever)
        
        results.append({
            "Retriever": type(retriever).__name__,
            "Hit Rate": hit_rate,
            "MRR": mrr
        })

    results_df = pd.DataFrame(results)

    return results_df


def compare_embeddings(data,embed_model_list,llm,**kwargs):
    """
    Compares multiple embedding models using same retriever on a QA dataset, evaluating their hit rate and MRR.

    Parameters:
        data (List[TextNode]): The list of TextNodes to generate the QA dataset from.
        llm (BaseLLMModel): The language model used to generate the QA dataset.
        retrievers_list (List[BaseEmbeddings]): A list of retriever instances to evaluate.

    Returns:
        pandas.DataFrame: A DataFrame containing the hit rate and MRR for each retriever.
    """
    qa_dataset = generate_qa_dataset(llm, data)
    
    results = []

    for embed_model in embed_model_list:
        retriever = NormalRetriever(data,embed_model,**kwargs)
        hit_rate, mrr = evaluate_from_dataset(qa_dataset, retriever)
        
        results.append({
            "Embeddings model": type(embed_model).__name__,
            "Hit Rate": hit_rate,
            "MRR": mrr
        })

    results_df = pd.DataFrame(results)

    return results_df