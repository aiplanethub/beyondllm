from .base import BaseLLMModel, ModelConfig
from typing import Any, Dict, Optional
from pydantic import Field
from dataclasses import dataclass

@dataclass
class HuggingFaceHubModel:
    """
    Class representing a Language Model (LLM) model using Hugging Face Hub.
    Example: 
    from enterprise_rag.llms import HuggingFaceHubModel
    llm = HuggingFaceHubModel(model="huggingfaceh4/zephyr-7b-alpha",token="<replace_with_your_token>",model_kwargs={"max_new_tokens":512,"temperature":0.1})
    """
    token: str
    model: str = "HuggingFaceh4/zephyr-7b-alpha"
    model_kwargs: Optional[Dict] = None

    def __post_init__(self):
        self.load_llm()

    def load_llm(self):
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError("HuggingFace Hub library is not installed. Please install it with `pip install huggingface_hub`.")
        
        try:
            self.client = InferenceClient(
                model = self.model,
                token = self.token,
            )
        except Exception as e:
            raise Exception("Failed to load the model from Hugging Face Hub:", str(e))

    def predict(self, prompt: Any):
        if self.client is None:
            raise ValueError("Model is not loaded. Please call load_llm() to load the model before making predictions.")
        
        return self.client.text_generation(prompt,**self.model_kwargs)

    @staticmethod
    def load_from_kwargs(kwargs): 
        model_config = ModelConfig(**kwargs)
        self.config = model_config
        self.load_llm()