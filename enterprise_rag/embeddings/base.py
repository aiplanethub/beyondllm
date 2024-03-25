import os
import yaml

class BaseEmbeddings:
    def __init__(self, model_name="AzureOpenAI",config_file=None,**kwargs):
        self.model_name = model_name
        
        if config_file:
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
                self.model_name = config.get('model_name', self.model_name)
                self.azure_endpoint = config.get('azure_endpoint', self.azure_endpoint)
                self.azure_key = config.get('azure_key', self.azure_key)
                self.api_version = config.get('api_version', self.api_version)

    def load(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def embed_text(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
