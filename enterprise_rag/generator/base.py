from pydantic import BaseModel
from typing import Any, Dict

class GeneratorConfig(BaseModel):
    """
    Base configuration model for all the configuration for Generator class.
    """
    pass

class BaseGenerator(BaseModel):
    """
    Base class for Language Model (LLM) models.
    """
    def invoke(self,question):
        raise NotImplementedError

    def evaluate(self,question):
        raise NotImplementedError