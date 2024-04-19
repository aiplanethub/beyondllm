from beyondllm.llms.base import BaseLLMModel, ModelConfig
from typing import Any, Dict, List, Optional
from dataclasses import dataclass,field
import os

@dataclass
class GeminiModel:
    """
    Class representing a Language Model (LLM) model using Google Generative AI
    Example:
    ```
    >>> from beyondllm.llms import GeminiModel
    >>> llm = GeminiModel(model_name="gemini-pro",google_api_key = "<your_api_key>",model_kwargs={"temperature":0.2})
    ```
    or 
    ```
    >>> import os
    >>> os.environ['GOOGLE_API_KEY'] = "***********" #replace with your key
    >>> from beyondllm.llms import GeminiModel
    >>> llm = GeminiModel(model_name="gemini-pro")
    ```
    """
    google_api_key:str = ""
    model_name:str = "gemini-pro"
    model_kwargs: dict = field(default_factory=lambda: {
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": 1,
                    "max_output_tokens": 2048,
            })

    def __post_init__(self):
        if not self.google_api_key:  
            self.google_api_key = os.getenv('GOOGLE_API_KEY') 
            if not self.google_api_key: 
                raise ValueError("GOOGLE_API_KEY is not provided and not found in environment variables.")
        self.load_llm()

    def load_llm(self):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Google Generative AI library is not installed. Please install it with ``pip install google-generativeai``.")
        
        try:
            VALID_MODEL_SUPPORT = ["gemini-1.0-pro","gemini-pro",""]
            if self.model_name not in VALID_MODEL_SUPPORT:
                raise "Model not supported. Currently we only support `gemini-pro` and `gemini-1.0-pro`"
            
            
            genai.configure(api_key = self.google_api_key)
            self.client = genai.GenerativeModel(model_name=self.model_name,
                                                generation_config=self.model_kwargs)

        except Exception as e:
            raise Exception("Failed to load the model from Gemini Google Generative AI:", str(e))


    def predict(self,prompt:Any):
        response = self.client.generate_content(prompt)
        return response.text
    
    @staticmethod
    def load_from_kwargs(self,kwargs): 
        model_config = ModelConfig(**kwargs)
        self.config = model_config
        self.load_llm()