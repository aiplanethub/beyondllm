from beyondllm.llms.base import BaseLLMModel, ModelConfig
from typing import Any, Dict, List, Optional
import os
from dataclasses import dataclass,field

@dataclass
class GroqModel:
    """
    Class representing a Chat Language Model (LLM) model using OpenAI
    Example:
    from beyondllm.llms import GroqModel
    llm = ChatOpenAIModel(model="gpt-3.5-turbo",api_key = "",model_kwargs = {"max_tokens":512,"temperature":0.1})
    """
    groq_api_key: str = ""
    model: str = field(default="mixtral-8x7b-32768")
    system_prompt: str = field(default="You are an AI assistant")

    def __post_init__(self):
        if not self.groq_api_key:  
            self.groq_api_key = os.getenv('GROQ_API_KEY') 
            if not self.groq_api_key: 
                raise ValueError("GROQ_API_KEY is not provided and not found in environment variables.")
        self.load_llm()

    def load_llm(self):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Groq library is not installed. Please install it with ``pip install groq``.")
        
        try:
            self.client = Groq(api_key=self.groq_api_key)
    
        except Exception as e:
            raise Exception("Failed to load the model from Groq:", str(e))
        
        return self.client

    def predict(self,prompt:Any):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"{self.system_prompt}",
                },
                {
                    "role": "user",
                    "content": f"{prompt}",
                },
            ],
            model=self.model,
        )
        return response.choices[0].message.content

    @staticmethod
    def load_from_kwargs(self,kwargs): 
        model_config = ModelConfig(**kwargs)
        self.config = model_config
        self.load_llm()