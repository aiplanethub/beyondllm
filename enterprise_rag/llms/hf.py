from .base import BaseLLMModel, ModelConfig
from typing import Any, Dict, Optional

class HuggingFaceHubModel(BaseLLMModel):
    """
    Class representing a Language Model (LLM) model using Hugging Face Hub.
    Example: 
    from enterprise_rag.llms import HuggingFaceHubModel
    llm = HuggingFaceHubModel(model="huggingfaceh4/zephyr-7b-alpha",token="<replace_with_your_token>",model_kwargs={"max_new_tokens":512,"temperature":0.1})
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
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError("HuggingFace Hub library is not installed. Please install it with `pip install huggingface_hub`.")
        
        try:
            self.llm = InferenceClient(
                model = self.config.model,
                token = self.config.token
            )
        except Exception as e:
            raise Exception("Failed to load the model from Hugging Face Hub:", str(e))

    def predict(self, prompt: Any):
        if self.llm is None:
            raise ValueError("Model is not loaded. Please call load_llm() to load the model before making predictions.")
        
        if self.config.model_kwargs is not None:
            return self.llm.text_generation(prompt, **self.config.model_kwargs)
        else:
            return self.llm.text_generation(prompt)


        