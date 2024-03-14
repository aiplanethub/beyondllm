from .embeddings.base import BaseEmbeddings

def get_embedding(embedding_type='HuggingFaceInferenceAPIEmbeddings', config_file=None, **kwargs):
    
    if embedding_type == 'HuggingFaceInferenceAPIEmbeddings':
        embedder = BaseEmbeddings(config_file=config_file, **kwargs)
    else:
        raise NotImplementedError(f"Embedder for the type '{embedding_type}' is not implemented.")
    
    return embedder

def evaluate_embeddings(config_file=None, **kwargs):
    
    pass