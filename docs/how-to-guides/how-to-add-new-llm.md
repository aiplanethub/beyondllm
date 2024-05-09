# ➕ How to add new LLM?

We're living in a time where there are many Large language models available. Each possessing its distinct characteristics. To integrate these models effectively, one must adhere to three common practices:

1. Configuring parameters required by the LLM, such as API Key and model name.
2. Loading the LLM through appropriate function calls.
3. Predicting responses from the LLM based on user prompts.

We follow the same procedures for integrating new LLMs. Here's an example of how to add a new LLM, such as Gemini.&#x20;

{% hint style="info" %}
Note:&#x20;

Each LLM has its own documentation. We should refer to their documentation to learn how to initialise and make predictions with them.
{% endhint %}

### Configure Parameters

As mentioned, each LLM model requires certain user inputs and configurations for integration. We define a dataclass that encapsulates the possible parameters needed to configure the LLM. By using a dataclass, we can implement a static method to load the keyword arguments. In the `__post_init__` function, we typically initialize the model. If the model requires an API key to be read from an environment variable, this can be declared in the `__post_init__` function. This format is consistent across other models such as `ChatOpenAI` and `HuggingFace`.

{% hint style="info" %}
Note: `load_from_kwargs` is a default static method, that is used in every LLM. This ensures that the user can enter any parameters that is supported by that model.&#x20;
{% endhint %}

```python
import os
from .base import BaseLLMModel, ModelConfig
from dataclasses import dataclass

@dataclass
class GeminiModel:
    """
    Class representing a Language Model (LLM) model using Google Generative AI
    Example:
    from enterprise-rag.llms import GeminiModel
    llm = GeminiModel(model_name="gemini-pro",google_api_key = "<your_api_key>")
    or 
    import os
    os.environ['GOOGLE_API_KEY'] = "***********" #replace with your key
    from enterprise-rag.llms import GeminiModel
    llm = GeminiModel(model_name="gemini-pro")
    """
    google_api_key:str = ""
    model_name:str = "gemini-pro"

    def __post_init__(self):
        if not self.google_api_key:  
            self.google_api_key = os.getenv('GOOGLE_API_KEY') 
            if not self.google_api_key: 
                raise ValueError("GOOGLE_API_KEY is not provided and not found in environment variables.")
        self.load_llm()
    
    @staticmethod
    def load_from_kwargs(self,kwargs): 
        model_config = ModelConfig(**kwargs)
        self.config = model_config
        self.load_llm()
```

### Load LLM

At Enterprise RAG we provide the flexibility to use any LLM based on user choice, this is where we provide the flexibility to add new LLM, and install only when they use it. The load LLM is a simple function that just initialize the model.&#x20;

```python
def load_llm(self):
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Google Generative AI library is not installed. Please install it with ``pip install google-generativeai``.")
    
    try:
        VALID_MODEL_SUPPORT = ["gemini-1.0-pro","gemini-pro"]
        if self.model_name not in VALID_MODEL_SUPPORT:
            raise "Model not supported. Currently we only support `gemini-pro` and `gemini-1.0-pro`"
        
        genai.configure(api_key = self.google_api_key)
        self.client = genai.GenerativeModel(model_name=self.model_name)

    except Exception as e:
        raise Exception("Failed to load the model from Gemini Google Generative AI:", str(e))

```

### predict&#x20;

The `predict` function takes user input and generates the response. Here, various API formats are used to generate the response, and we select the index where the complete response is displayed.

```python
def predict(self,prompt:Any):
    response = self.client.generate_content(prompt)
    return response.text
```
