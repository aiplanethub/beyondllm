from .base import BaseEmbeddings
import subprocess
import sys
try:
    from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
except ImportError:
    user_agree = input("The feature you're trying to use requires library 'llama_index.embeddings.huggingface'. Would you like to install it now? [y/N]: ")
    if user_agree.lower() == 'y':
        subprocess.check_call([sys.executable, "-m", "pip", "install", "llama_index.embeddings.huggingface"])
        from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
    else:
        raise ImportError("The required 'llama-index-embeddings-huggingface_library' is not installed.")
import os

class HuggingFaceInferenceEmbedding(BaseEmbeddings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = kwargs.get('model_name')

    def load(self):
        embed_model = HuggingFaceInferenceAPIEmbedding(model_name=self.model_name)

        return embed_model