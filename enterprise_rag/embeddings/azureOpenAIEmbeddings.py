from .base import BaseEmbeddings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

import os

class AzureEmbeddings(BaseEmbeddings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.azure_endpoint = kwargs.get('azure_endpoint', os.getenv('AZURE_EMBEDDINGS_ENDPOINT'))
        self.azure_key = kwargs.get('api_key', os.getenv('AZURE_EMBEDDINGS_KEY'))
        self.api_version = kwargs.get('api_version')
        self.deployment_name = kwargs.get('deployment_name')

    def load(self):
        embed_model = AzureOpenAIEmbedding(
            api_key=self.azure_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
            deployment_name=self.deployment_name
        )
        return embed_model
    