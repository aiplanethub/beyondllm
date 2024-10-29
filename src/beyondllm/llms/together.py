from beyondllm.llms.base import BaseLLMModel, ModelConfig
from typing import Any, Dict
from dataclasses import dataclass, field
import os

@dataclass
class TogetherModel:
    """
    Class representing a Language Model (LLM) using Together AI.

    Example:
    ```
    >>> llm = TogetherModel(api_key="<your_api_key>", model_kwargs={"temperature": 0.7})
    ```
    or
    ```
    >>> import os
    >>> os.environ['TOGETHER_API_KEY'] = "***********" #replace with your key
    >>> llm = TogetherModel()
    ```
    """
    api_key: str = " "
    model_name: str = "meta-llama/Llama-3-8b-chat-hf"
    model_kwargs: dict = field(default_factory=lambda: {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 1024,
    })

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv('TOGETHER_API_KEY')
            if not self.api_key:
                raise ValueError("TOGETHER_API_KEY is not provided and not found in environment variables.")
        self.load_llm()

    def load_llm(self):
        try:
            from together import Together
        except ImportError:
            print("The together module is not installed. Please install it with 'pip install together'.")
        
        try:
            self.client = Together(api_key=self.api_key)
        except Exception as e:
            raise Exception(f"Failed to initialize Together client: {str(e)}")

    def predict(self, prompt: Any) -> str:
        """Generate a response from the model based on the provided prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": "You are a highly skilled software engineer. Provide detailed explanations and code examples when relevant."},
                          {"role": "user", "content": prompt}]
                
            )
            response_text = response.choices[0].message.content
            return response_text
           
        except Exception as e:
            raise Exception(f"Failed to generate prediction: {str(e)}")

    @staticmethod
    def load_from_kwargs(self, kwargs: Dict):
        model_config = ModelConfig(**kwargs)
        self.config = model_config
        self.load_llm()

