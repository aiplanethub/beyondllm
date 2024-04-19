from beyondllm.llms.base import BaseLLMModel, ModelConfig
import os
from typing import Any, Dict, Optional
from pydantic import Field
from dataclasses import dataclass,field

@dataclass
class HuggingFaceHubModel:
    """
    Class representing a Language Model (LLM) model using Hugging Face Hub.
    ### Example: 
    ```
    >>> from beyondllm.llms import HuggingFaceHubModel
    >>> llm = HuggingFaceHubModel(model="huggingfaceh4/zephyr-7b-alpha",token="<replace_with_your_token>",model_kwargs={"max_new_tokens":512,"temperature":0.1})
    ```
    or 
    ```
    >>> import os
    >>> os.environ['HUGGINGFACE_ACCESS_TOKEN'] = "hf_*************"
    >>> from beyondllm.llms import HuggingFaceHubModel
    >>> llm = HuggingFaceHubModel(model="HuggingFaceh4/zephyr-7b-beta",model_kwargs={"max_new_tokens":512,"temperature":0.1})
    ```
    """
    
    token: str = ""
    model: str = "HuggingFaceh4/zephyr-7b-beta"
    model_kwargs: dict = field(default_factory=lambda: {"max_new_tokens":1024,"temperature": 0.1,"top_p":0.95,"repetition_penalty": 1.1,"return_full_text": False})

    def __post_init__(self):
        if not self.token:  
            self.token = os.getenv('HUGGINGFACE_ACCESS_TOKEN') 
            if not self.token: 
                raise ValueError("HUGGINGFACE_ACCESS_TOKEN is not provided and not found in environment variables.")
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
    def load_from_kwargs(self,kwargs): 
        model_config = ModelConfig(**kwargs)
        self.config = model_config
        self.load_llm()