from .base import BaseLLMModel
from typing import Any, Dict, List, Mapping, Optional

class HuggingFaceHubModel(BaseLLMModel):
    """
    Example

    from enterprise-rag.llms import HuggingFaceHubModel
    llm = HuggingFaceHubModel(model="zephyr-7b-beta",token="<your-hf-token>")
    """
    def validate_env(self):
        try:
            from huggingface_hub import InferenceClient

            hf_repo_id: Optional[str] = None
            model_kwargs: Dict = None
            huggingface_access_token: Optional[str] = None

            client = InferenceClient(model=hf_repo_id,token=huggingface_access_token,model_kwargs=model_kwargs)

        except ImportError:
            raise ValueError(
                "Couldn't find HuggingFace Hub library"
                "Please install it with `pip install huggingface_hub`"
            )

    def load_llm(self):
        pass
