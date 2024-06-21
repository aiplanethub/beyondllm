from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .base import BaseMemory, MemoryConfig
import re

@dataclass
class ChatBufferMemory(BaseMemory):
    """
    ChatBufferMemory stores the recent conversation history as a buffer.

    Args:
        window_size (int): The maximum number of messages to store in the buffer.
        memory_key (str): The key to identify the memory.
        max_tokens (int): The maximum number of tokens to store in the buffer. Defaults to None, which means no limit. 
    """

    window_size: int = 5
    memory_key: str = "chat_buffer"
    _buffer: List[Dict[str, str]] = field(default_factory=list)
    config: MemoryConfig = field(default_factory=MemoryConfig)

    def load_memory(self, **kwargs) -> None:
        self._buffer = []
        self.config = MemoryConfig(**kwargs)

    def add_to_memory(self, question: str, response: str) -> None:
        """Adds the context and response to the buffer, maintaining the window size."""
        self._buffer.append(
            {"question": question, "response": response}
        )

        if len(self._buffer) > self.window_size+1:
            self._buffer.pop(0)
    
    def get_memory(self) -> List[Dict[str, str]]:
        """Returns the current buffer."""
        return self._buffer

    def clear_memory(self) -> None:
        """Clears the buffer."""
        self._buffer = []