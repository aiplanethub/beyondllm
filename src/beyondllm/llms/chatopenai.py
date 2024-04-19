from beyondllm.llms.base import BaseLLMModel, ModelConfig
from typing import Any, Dict, List, Optional
import os
from dataclasses import dataclass,field

@dataclass
class ChatOpenAIModel:
    """
    Class representing a Chat Language Model (LLM) model using OpenAI
    Example:
    from beyondllm.llms import ChatOpenAIModel
    llm = ChatOpenAIModel(model="gpt-3.5-turbo",api_key = "",model_kwargs = {"max_tokens":512,"temperature":0.1})
    """
    api_key: str = ""
    model: str = field(default="gpt-3.5-turbo")
    model_kwargs: Optional[Dict]  = None

    def __post_init__(self):
        if not self.api_key:  
            self.api_key = os.getenv('OPENAI_API_KEY') 
            if not self.api_key: 
                raise ValueError("OPENAI_API_KEY is not provided and not found in environment variables.")
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
    def load_from_kwargs(self,kwargs): 
        model_config = ModelConfig(**kwargs)
        self.config = model_config
        self.load_llm()