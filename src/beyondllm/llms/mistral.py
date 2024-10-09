from beyondllm.llms.base import BaseLLMModel, ModelConfig
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import os

@dataclass
class MistralModel(BaseLLMModel):
    """
    Class representing a Mistral Language Model (LLM) using Mistral AI API.
    Example:
    >>> from beyondllm.llms import MistralModel
    >>> llm = MistralModel(model_name="mistral-medium", api_key="<your_api_key>", model_kwargs={"max_tokens": 512, "temperature": 0.7})
    """
    api_key: str = ""
    model_name: str = field(default="mistral-medium")
    model_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.7,
        "max_tokens": 512,
    })
    client: Any = None

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv('MISTRAL_API_KEY')
            if not self.api_key:
                raise ValueError("MISTRAL_API_KEY is not provided and not found in environment variables.")
        self.load_llm()

    def load_llm(self):
        try:
            from mistralai.client import MistralClient
            from mistralai.models.chat_completion import ChatMessage
        except ImportError:
            raise ImportError("Mistral AI library is not installed. Please install it with `pip install mistralai`.")
        
        try:
            self.client = MistralClient(api_key=self.api_key)
            self.ChatMessage = ChatMessage
        except Exception as e:
            raise Exception(f"Failed to initialize Mistral AI client: {str(e)}")

    def predict(self, prompt: Any) -> str:
        if not self.client:
            raise Exception("Mistral AI client is not initialized.")
        try:
            messages = [self.ChatMessage(role="user", content=prompt)]
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                **self.model_kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")

    @staticmethod
    def load_from_kwargs(kwargs: Dict[str, Any]) -> 'MistralModel':
        model_config = ModelConfig(**kwargs)
        return MistralModel(**model_config.__dict__)