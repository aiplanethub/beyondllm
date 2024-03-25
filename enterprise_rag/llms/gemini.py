from typing import Any, Dict, List, Optional
from .base import BaseLLMModel, ModelConfig

class GeminiModel(BaseLLMModel):
    """
    Class representing a Language Model (LLM) model using Google Generative AI
    Example:
    from enterprise-rag.llms import GeminiModel
    llm = GeminiModel(model_name="gemini-pro",google_api = "<your_api_key>")
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
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Google Generative AI library is not installed. Please install it with ``pip install google-generativeai``.")
        
        try:
            VALID_MODEL_SUPPORT = ["gemini-1.0-pro","gemini-pro"]
            if self.config.model not in VALID_MODEL_SUPPORT:
                raise "Model not supported. Currently we only support `gemini-pro` and `gemini-1.0-pro`"
            
            genai.configure(api_key = self.config.google_api_key)
            self.llm = genai.GenerativeModel(model_name=self.config.model)

        except Exception as e:
            raise Exception("Failed to load the model from Gemini Google Generative AI:", str(e))


    def predict(self,prompt:Any):
        response = self.llm.generate_content(prompt)
        return response.text