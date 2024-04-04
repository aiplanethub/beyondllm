from pydantic import BaseModel
from typing import Any, Optional, Dict

class ModelConfig(BaseModel):
    """Base configuration model for all LLMs.

    This class can be extended to include more fields specific to certain LLMs.
    """
    pass 


class BaseLLMModel(BaseModel):
    """
    Base class for Language Model (LLM) models.
    """

    def load_llm(self):
        """
        Method to load the language model.
        """
        raise NotImplementedError
    

    def predict(self, query: Any):
        """
        Method to generate predictions from the loaded language model.
        
        Parameters:
            query (Any): Input query for prediction.
        
        Returns:
            Any: Model prediction for the given input query.
        """
        raise NotImplementedError