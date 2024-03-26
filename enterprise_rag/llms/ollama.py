from typing import Any, Dict, List, Mapping, Optional
    
from .base import BaseLLMModel, ModelConfig
from typing import Any, Dict, Optional

class OllamaModel(BaseLLMModel):
    """
    Ollama locally run large language models.

    Example: 
    from enterprise_rag.llms import OllamaModel
    llm = Ollama(model="llama2")
    
    """
    def __init__(self, config: ModelConfig):
        """
        Initializes the HuggingFaceHubModel with a ModelConfig object.
        
        Parameters:
            config (ModelConfig): Configuration parameters for the model.
        """
        super().__init__()
        self.config = config
        
    def load_llm(self):
        try:
            import ollama
            
            self.client = ollama.Client(host='http://localhost:11434')

        except ImportError:
            raise ImportError("Ollama library is not installed. Please install it with `pip install ollama`.")
        except ConnectionError:
            raise ConnectionError("Ollama by default runs on LocalHost:11434")

    def predict(self, prompt: Any):
        try:
            response = client.chat(
                model= self.config.model,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                stream=True)
        except ollama.ResponseError:
            raise ollama.ResponseError("You need to pull model first. ollama pull <model-name>")
        return response