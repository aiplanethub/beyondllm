from .embeddings.azureOpenAIEmbeddings import AzureEmbeddings
from .embeddings.hf import HuggingFaceEmbeddings
from .embeddings.inference import HuggingFaceInferenceEmbedding
# from .embeddings.evaluate import evaluateEmbeddings

def get_embed_model(model_name="AzureOpenAI", **kwargs):
    if model_name.startswith("HF-"):
        parts = model_name.split("-", 1)
        if len(parts) > 1:
            hf_model_name = parts[1]
        else:
            raise ValueError("Huggingface model name required after HF- prefix")
        embedder = HuggingFaceEmbeddings(hf_model_name)        
    elif model_name.startswith("HFI-"):
        parts = model_name.split("-", 1)
        if len(parts) > 1:
            hf_model_name = parts[1]
        else:
            raise ValueError("Huggingface Inference model name required after HFI- prefix")
        embedder = HuggingFaceInferenceEmbedding(hf_model_name)        
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
