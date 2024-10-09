from beyondllm.llms.base import BaseLLMModel, ModelConfig
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import os
import base64
@dataclass
class MistralModel:
    """
    Class representing a Mistral Language Model (LLM) using Mistral AI API.
    Example:
    >>> from beyondllm.llms import MistralModel
    >>> llm = MistralModel(model_name="mistral-medium", api_key="<your_api_key>", model_kwargs={"max_tokens": 512, "temperature": 0.7})
    """
    api_key: str = ""
    model_name: str = "mistral-large-latest"

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv('MISTRAL_API_KEY')
            if not self.api_key:
                raise ValueError("MISTRAL_API_KEY is not provided and not found in environment variables.")
        self.load_llm()

    def load_llm(self):
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError("Mistral AI library is not installed. Please install it with `pip install mistralai`.")
        
        try:
            self.client = Mistral(api_key=self.api_key)
        except Exception as e:
            raise Exception(f"Failed to initialize Mistral AI client: {str(e)}")
        return self.client
    
    def _process_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def predict(self, prompt: Any, image_path: str = None) -> str:
        if image_path:
            media_type = image_path.split(".")[-1]
            try:
                if image_path.startswith("http"):
                    image_url = image_path
                elif media_type in ['jpg','jpeg','png','webp','aiff']:
                    base64_image = self._process_image(image_path)
                    image_url = f"data:image/{media_type};base64,{base64_image}" 
                
                messages = [
                        {"role": "user","content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": image_url
                                }
                            ]
                        }
                    ]
                response = self.client.chat.complete(
                    model = self.model_name,
                    messages = messages
                )
            except Exception as e:
                raise Exception(f"Failed to recognize: {e}")
        else:
            response = self.client.chat.complete(
                model = self.model_name,
                messages=[{"role":"user", "content":prompt}]
            )
        
        return response.choices[0].message.content

    @staticmethod
    def load_from_kwargs(self,kwargs): 
        model_config = ModelConfig(**kwargs)
        self.config = model_config
        self.load_llm()