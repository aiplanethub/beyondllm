from beyondllm.llms.base import BaseLLMModel, ModelConfig
from typing import Any, Dict, List, Optional
from dataclasses import dataclass,field
from PIL import Image
import os

@dataclass
class GeminiMultiModal:
    """
    Class representing a Language Model (LLM) model using Google Generative AI
    Example:
    ```
    >>> from beyondllm.llms import GeminiMultiModal
    >>> llm = GeminiMultiModal(model_name="gemini-pro-vision",google_api_key = "<your_api_key>",model_kwargs={"temperature":0.2})
    ```
    or 
    ```
    >>> import os
    >>> os.environ['GOOGLE_API_KEY'] = "***********" #replace with your key
    >>> from beyondllm.llms import GeminiMultiModal
    >>> llm = GeminiMultiModal(model_name="gemini-pro-vision")
    ```
    """
    google_api_key:str = ""
    model_name:str = "gemini-pro-vision"
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
            VALID_MODEL_SUPPORT = ["gemini-1.0-pro-vision-latest","gemini-pro-vision"]
            if self.model_name not in VALID_MODEL_SUPPORT:
                raise f"Model not supported. Currently we only support: {','.join(VALID_MODEL_SUPPORT)}."
            
            genai.configure(api_key = self.google_api_key)
            self.client = genai.GenerativeModel(model_name=self.model_name,
                                                generation_config=self.model_kwargs)

        except Exception as e:
            raise Exception("Failed to load the model from Gemini Google Generative AI:", str(e))

    def predict(self,image: Image ,prompt:Optional[str] = None):
        if prompt:
            response = self.client.generate_content([prompt,image])
        else:
            response = self.client.generate_content(image)
        return response.text
    
    @staticmethod
    def load_from_kwargs(self,kwargs): 
        model_config = ModelConfig(**kwargs)
        self.config = model_config
        self.load_llm()
