from typing import Any, Dict
from dataclasses import dataclass, field
import os

@dataclass
class CerebrasModel:
    """
    Class representing a Language Model (LLM) model using Cerebras AI.
    Example:
    ```
    >>> from beyondllm.llms import CerebrasModel
    >>> llm = CerebrasModel(model_name="llama3.1-8b", model_kwargs={"temperature":0.2, "max_completion_tokens":1024})
    ```
    or
    ```
    >>> import os
    >>> os.environ['CEREBRAS_API_KEY'] = "***********"
    >>> from beyondllm.llms import CerebrasModel
    >>> llm = CerebrasModel()
    ```
    """
    api_key: str = ""
    model_name: str = "llama3.1-8b"
    model_kwargs: dict = field(default_factory=lambda: {
                    "stream": False,
                    "temperature": 0.2,
                    "top_p": 1,
                    "max_completion_tokens": 2048
    })

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("CEREBRAS_API_KEY")
            if not self.api_key:
                raise ValueError("CEREBRAS_API_KEY is not provided and not found in environment variables.")
        self.load_llm()

    def load_llm(self):
        try:
            from cerebras.cloud.sdk import Cerebras
        except ImportError:
            raise ImportError("Cerebras library is not installed. Please install it with `pip install cerebras-cloud-sdk`.")

        try:
            self.client = Cerebras(api_key=self.api_key)
        except Exception as e:
            raise Exception(f"Failed to initialize Cerebras client: {str(e)}")

    def predict(self, prompt: Any) -> str:
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": str(prompt)},
                ],
                model=self.model_name,
                **self.model_kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Failed to generate prediction: {str(e)}")

    @staticmethod
    def load_from_kwargs(self, kwargs: Dict):
        model_config = ModelConfig(**kwargs)
        self.config = model_config
        self.load_llm()