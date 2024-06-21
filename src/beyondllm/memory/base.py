from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from pydantic import BaseModel

class MemoryConfig(BaseModel):
    """Base configuration model for all memory components.

    This class can be extended to include more fields specific to certain memory components.
    """
    pass 

class BaseMemory(ABC):
    """
    Base class for memory components.
    """

    @abstractmethod
    def load_memory(self, **kwargs) -> None:
        """Loads the memory component."""
        raise NotImplementedError

    @abstractmethod
    def add_to_memory(self,  context: str, response: str) -> None:
        """Adds the conversation context and response to memory."""
        raise NotImplementedError

    @abstractmethod
    def get_memory(self) -> Any:
        """Returns the current memory."""
        raise NotImplementedError

    @abstractmethod
    def clear_memory(self) -> None:
        """Clears the memory."""
        raise NotImplementedError
