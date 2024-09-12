import streamlit as st

# Main content section
st.title("Perform quick LLM evaluations🔄️")

st.write("Unlock the power of language models quickly and efficiently with our streamlined evaluation process. Whether you’re working with a single LLM or comparing multiple models, our platform makes it easy to assess Context Relevance, Answer Relevance, and Groundedness for your needs.")

st.html("""
<ul>
  <li><strong>Quick and Simple:</strong> Perform rapid evaluations with a few clicks. No need for complex setups or lengthy processes.</li>
  <li><strong>Single Model Assessments:</strong> Evaluate one language model at a time to get detailed insights for complete visibility.</li>
  <li><strong>Multiple Model Comparisons:</strong> Simultaneously compare multiple LLMs to see how they stack up against each other, saving you time and effort.</li>
</ul>
""")

st.write("Start your evaluations now and find the language model that fits your project perfectly.")

st.header("Release Notes:")

st.markdown("""
  <style>
    div[data-testid="stExpander"] > details > summary > span > div > p {
      font-size: 1.1rem;
    }
  </style>"""
, unsafe_allow_html=True)

versions = [
  {
    "version_name": "v1.1.0",
    "header": "Simultaneous Evaluation with Multiple LLMs Support",
    "release_date": "August 29, 2024",
    "markdown_content": """
      This release introduces a major update to our Streamlit app, now featuring the capability to perform evaluations with multiple LLMs simultaneously. Users can now select and evaluate multiple models in one go, expanding the flexibility and depth of evaluations. 
      
      **:blue[<u>What's New:</u>]**
        - <u>:rainbow[**Multiple LLM Evaluations Simultaneously**:]</u> :green[Evaluate multiple models in one go, providing a more comprehensive analysis.]
        - <u>:rainbow[**Color Coding**:]</u> :green[Selected LLM models are now highlighted with background colors corresponding to their providers. In order to differentiate the models from different LLM provides during Multiple LLM Evaluations.]

      **<u>Supported Sources:</u>**
      - URL
      - YouTube
      - PDF
      - DOCX

      **<u>Supported LLMs:</u>**
      - **Gemini**
          - gemini-1.0-pro
          - gemini-pro
          - gemini-1.5-pro-latest
      - **OpenAI**
          - gpt-3.5-turbo
          - gpt-4
          - gpt-4-turbo
          - gpt-3.5-turbo-16k
      - **Azure OpenAI**
          - gpt-35-turbo
          - gpt-4
          - gpt-35-turbo-16k
      - **Anthropic**
          - claude-3-5-sonnet-20240620
          - claude-3-haiku-20240307
          - claude-3-sonnet-20240229
          - claude-3-opus-20240229
      - **Groq**
          - mixtral-8x7b-32768
          - gemma2-9b-it
          - llama-3.1-8b-instant
          - llama3-70b-8192
          - llama3-8b-8192
          - llama3-groq-70b-8192-tool-use-preview
          - llama3-groq-8b-8192-tool-use-preview
      - **Hugging Face**
          - huggingfaceh4/zephyr-7b-alpha
          - huggingfaceh4/zephyr-7b-beta

      **<u>Evaluation Metrics:</u>**
      - Context Relevancy
      - Answer Relevancy
      - Groundedness

      This release enhances the app's functionality by allowing simultaneous evaluations across multiple LLMs, providing a more comprehensive analysis. Future updates will continue to focus on improving the user experience and adding new features.

      **Full Changelog**: https://github.com/ritwickbhargav80/quick-llm-model-evaluations/commits/v1.1.0
    """
},
    {
        "version_name": "v1.0.0",
        "header": "Single LLM Model Evaluation with BeyondLLM Latest",
        "release_date": "August 28, 2024",
        "markdown_content": """
          This release introduces the initial version of our Streamlit app for single-model evaluations using beyondllm. Users need to provide input such as source related information, and credentials for large language models (LLMs). The app also supports Retrieval-Augmented Generation (RAG) and provides comprehensive evaluation metrics.

            **<u>Supported Sources:</u>**
            - URL
            - YouTube
            - PDF
            - DOCX

            **<u>Supported LLMs:</u>**
            - Gemini
                - gemini-1.0-pro
                - gemini-pro
                - gemini-1.5-pro-latest
            - OpenAI
                - gpt-3.5-turbo
                - gpt-4
                - gpt-4-turbo
            - Azure OpenAI
                - gpt-35-turbo
                - gpt-35-turbo-16k
                - gpt-4
            - Anthropic
                - claude-3-sonnet-20240229
                - claude-3-haiku-20240307
                - claude-3-opus-20240229
                - claude-3-5-sonnet-20240620

            **<u>Evaluation Metrics:</u>**
            - Context Relevancy
            - Answer Relevancy
            - Groundedness

            This release marks the first stable version of the app, tested for a smooth user experience across different models and sources. Future updates will include extended features and further optimizations.

            **Full Changelog**: https://github.com/ritwickbhargav80/quick-llm-model-evaluations/commits/v1.0.0
        """
    }
]

for idx, version in enumerate(versions):
  expander = st.expander(f"{version['version_name']}{' (Latest)' if idx == 0 else ''}")
  expander.subheader(version["header"])
  expander.write(f"Release date: {version['release_date']}")
  expander.markdown(version["markdown_content"], unsafe_allow_html=True)