from typing import Any, Dict, List, Optional
from pydantic import Field
from .base import BaseLLMModel, ModelConfig
from dataclasses import dataclass

@dataclass
class ChatOpenAIModel:
    """
    Class representing a Chat Language Model (LLM) model using OpenAI
    Example:
    from enterprise-rag.llms import ChatOpenAIModel
    llm = ChatOpenAIModel(model="gpt-3.5-turbo",api_key = "",model_kwargs = {"max_tokens":512,"temperature":0.1})
    """
    api_key: str
    model: str = "gpt4"
    model_kwargs: Optional[Dict]  = None

    def __post_init__(self):
        self.load_llm()

    def load_llm(self):
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI library is not installed. Please install it with ``pip install openai``.")
        
        try:
            self.client = openai.OpenAI(api_key=self.api_key)

        except Exception as e:
            raise Exception("Failed to load the model from OpenAI:", str(e))
        
        return self.client

    def predict(self,prompt:Any):
        if self.model_kwargs is not None:
            response = self.client.chat.completions.create(
                model = self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                **self.model_kwargs
            )
        else:
            response = self.client.chat.completions.create(
                model = self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
        
        return response.choices[0].message.content

    @staticmethod
    def load_from_kwargs(kwargs): 
        model_config = ModelConfig(**kwargs)
        self.config = model_config
        self.load_llm()