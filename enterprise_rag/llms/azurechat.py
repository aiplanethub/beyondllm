from typing import Any, Dict, List, Optional, Field
from .base import BaseLLMModel, ModelConfig

class AzureOpenAIModel(BaseLLMModel):
    """
    Class representing a Chat Language Model (LLM) model using OpenAI
    Example:
    from enterprise-rag.llms import ChatOpenAIModel
    llm = AzureOpenAIModel(model="gpt4",api_key = "<your_api_key>",deployment_name="",endpoint_url="",model_kwargs={"max_tokens":512,"temperature":0.1})
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
            import openai
        except ImportError:
            raise ImportError("AzureChat Model is not installed. Please install it with ``pip install openai``.")
        
        try:
            self.client = openai.AzureOpenAI(
                api_version = "2023-05-15",
                azure_endpoint = self.config.endpoint_url,
                azure_deployment = self.config.deployment_name,  
                api_key = self.config.api_key
            )
        except Exception as e:
            raise Exception("Failed to load the model from Azure OpenAI:", str(e))


    def predict(self,prompt:Any):
        try:
            if self.config.model_kwargs is not None:
                response = self.client.chat.completions.create(
                    **self.config.model_kwargs,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
            else:
                response = self.client.chat.completions.create(
                    self.config.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
        except NotFoundError:
            raise "Check out `https://platform.openai.com/docs/models/overview` to see the supported model from Azure OpenAI"
        
        return response.choices[0].message.content