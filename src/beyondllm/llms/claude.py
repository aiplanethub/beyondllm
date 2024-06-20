from beyondllm.llms.base import BaseLLMModel, ModelConfig
from typing import Any, Dict, List, Optional
import os
from dataclasses import dataclass,field

@dataclass
class ClaudeModel:
    """
    Class representing a Claude model from Antrophic
    Example:
    from beyondllm.llms import ClaudeModel
    llm = ClaudeModel(model="claude-3-5-sonnet-20240620",api_key = "",model_kwargs = {"max_tokens":512,"temperature":0.1})
    """
    api_key: str = ""
    model: str = field(default="claude-3-5-sonnet-20240620")
    model_kwargs: dict = field(default_factory=lambda: {
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": 1,
                    "max_tokens": 1000,
            })

    def __post_init__(self):
        if not self.api_key:  
            self.api_key = os.getenv('ANTHROPIC_API_KEY') 
            if not self.api_key: 
                raise ValueError("ANTHROPIC_API_KEY is not provided and not found in environment variables.")
        self.load_llm()

    def load_llm(self):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic library is not installed. Please install it with ``pip install anthropic``.")
        
        try:
            self.client = anthropic.Anthropic(api_key=self.api_key)

        except Exception as e:
            raise Exception("Failed to load the model from Anthropic:", str(e))
        
        return self.client

    def predict(self,prompt:Any):
        if self.model_kwargs is not None:
            response = self.client.messages.create(
                model = self.model,
                **self.model_kwargs,
                system="You are a world-class AI Assistant",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
        else:
            response = self.client.messages.create(
                model = self.model,
                system="You are a world-class AI Assistant",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
        
        return response.content[0].text
    
    @staticmethod
    def load_from_kwargs(self,kwargs): 
        model_config = ModelConfig(**kwargs)
        self.config = model_config
        self.load_llm()