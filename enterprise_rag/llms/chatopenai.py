from typing import Any, Dict, List, Optional
from .base import BaseLLMModel, ModelConfig

class ChatOpenAIModel(BaseLLMModel):
    """
    Class representing a Chat Language Model (LLM) model using OpenAI
    Example:
    from enterprise-rag.llms import ChatOpenAIModel
    llm = ChatOpenAIModel(model_name="gpt3.5-turbo",openai_api_key = "<your_api_key>")
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
            import openai
        except ImportError:
            raise ImportError("OpenAI library is not installed. Please install it with ``pip install openai``.")
        
        try:
            pass

        except Exception as e:
            raise Exception("Failed to load the model from OpenAI:", str(e))


    def predict(self,prompt:Any):
        response = self.llm.predict(prompt)
        return response