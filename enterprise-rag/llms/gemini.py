from typing import Any, Dict, List, Mapping, Optional

class GeminiModel(BaseLLMModel):
    """
    Example:
    from enterprise-rag.llms import GeminiModel
    llm = GeminiModel(model_name="gemini-pro",google_api = "<your_api_key>")
    """

    def validate_env(self):
        try:
            import google.generativeai as genai
            model_name: Optional[str] = None
            google_api_key: Optional[str] = None

            genai.configure(api_key = google_api_key)
            client =  genai.GenerativeModel(
                model_name=model_name
            )

        except ImportError:
            raise ValueError(
                "Couldn't find Gemini Model. This requires Google Generative support"
                "Please install it with `pip install google-generativeai`"
            )

    def load_llm(self):
        pass