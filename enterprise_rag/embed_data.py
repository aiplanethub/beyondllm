from .embeddings.azureOpenAIEmbeddings import AzureEmbeddings
from .embeddings.huggingFaceEmbeddings import HuggingFaceEmbedding
from .embeddings.evaluate import evaluateEmbeddings

def get_embed_model(model_name="AzureOpenAI", **kwargs):
    if model_name.startswith("HF-"):
        parts = model_name.split("-", 1)
        if len(parts) > 1:
            hf_model_name = parts[1]
        else:
            raise ValueError("Begin Huggingface models names with the HF- prefix")
        embedder = HuggingFaceEmbedding(hf_model_name)        
    elif model_name=='AzureOpenAI':
        embedder = AzureEmbeddings(**kwargs)
    else:
        raise NotImplementedError(f"Embedder for the model '{model_name}' is not implemented.")
    
    return embedder

# def evaluate_embeddings(embedding_models,nodes, llm):
#     # Placeholder for evaluation logic
#     print("Evaluating embeddings:")
#     results_df = evaluateEmbeddings(embedding_models,nodes,llm)
#     return results_df
