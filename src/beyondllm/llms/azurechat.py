from beyondllm.llms.base import BaseLLMModel, ModelConfig
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class AzureOpenAIModel:
    """
    Class representing a Chat Language Model (LLM) model using OpenAI
    Example:
    >>> from beyondllm.llms import ChatOpenAIModel
    >>> llm = AzureOpenAIModel(model="gpt4",api_key = "<your_api_key>",deployment_name="",endpoint_url="",model_kwargs={"max_tokens":512,"temperature":0.1})
    """
    api_key: str
    deployment_name: str
    endpoint_url: str
    model: str = "gpt4"
    model_kwargs: Optional[Dict] = None

    def __post_init__(self):
        self.load_llm()

    def load_llm(self):
        try:
            import openai
        except ImportError:
            raise ImportError("AzureChat Model is not installed. Please install it with ``pip install openai``.")
        
        try:
            self.client = openai.AzureOpenAI(
                api_version = "2023-05-15",
                azure_endpoint = self.endpoint_url,
                azure_deployment = self.deployment_name,  
                api_key = self.api_key
            )
        except Exception as e:
            raise Exception("Failed to load the model from Azure OpenAI:", str(e))


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