from beyondllm.llms.base import BaseLLMModel, ModelConfig
from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class OllamaModel:
    """
    Ollama locally run large language models.
    
    ## Example:
    ```
    >>> from beyondllm.llms import OllamaModel
    >>> llm = OllamaModel(model="llama2")
    ```
    """
    model: str

    def __post_init__(self):
        self.load_llm()
        
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
            import ollama
            response = self.client.chat(
                model= self.model,
                messages=[
                {
                    'role': 'system',
                    'content': f"You are an AI assitant that is using Ollama library. You are a {self.model} Large Language model that is the backbone that helps with user prompts."
                },
                {
                    'role': 'user',
                    'content': prompt
                }],
                stream=False)

        except ollama.ResponseError:
            raise ollama.ResponseError("You need to pull model first. ollama pull <model-name>")
        return response['message']['content']

    @staticmethod
    def load_from_kwargs(self,kwargs): 
        model_config = ModelConfig(**kwargs)
        self.config = model_config
        self.load_llm()