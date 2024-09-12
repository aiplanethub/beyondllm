# Perform Quick LLM Evaluations 🔄

Unlock the power of language models quickly and efficiently with our streamlined evaluation process. Whether you're working with a single LLM or comparing multiple models, our platform makes it easy to assess Context Relevance, Answer Relevance, and Groundedness for your needs.

## Features

- **Quick and Simple:** Perform rapid evaluations with a few clicks. No need for complex setups or lengthy processes.
- **Single Model Assessments:** Evaluate one language model at a time to get detailed insights for complete visibility.
- **Multiple Model Comparisons:** Simultaneously compare multiple LLMs to see how they stack up against each other, saving you time and effort.

Start your evaluations now and find the language model that fits your project perfectly.

### Supported Sources:
  - URL
  - YouTube
  - PDF
  - DOCX

### Supported LLMs:
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

### Evaluation Metrics:
  - Context Relevancy
  - Answer Relevancy
  - Groundedness

## Get Started

### **Step 1: Setting up the Application**

- Clone the repository:
    ```sh
    git clone https://github.com/aiplanethub/beyondllm.git
    cd quick-llm-model-evaluations
    ```

- (Optional) Create the Virtual Environment:
   ```sh
   python -m venv .venv
   ```

- (Optional) Activate the Virtual Environment:

   - On Windows:
     
       - With CMD:
       ```sh
       .\.venv\Scripts\activate.bat
       ```
       - With Powershell:
       ```sh
       .\.venv\Scripts\activate.ps1
       ```
       
   - On macOS/Linux:
     ```sh
     source .venv/Scripts/activate
     ```
    
- Ensure you have the following libraries installed:
  ```sh
  pip install beyondllm validators llama-index-readers-web youtube_transcript_api llama-index-readers-youtube-transcript llama-index-embeddings-azure_openai anthropic groq huggingface_hub llama-index-embeddings-fastembed llama-index-embeddings-huggingface docx2txt
  ```

- Launch the Streamlit application:
    ```sh
    streamlit run cookbook/Evaluating_LLM_Models/Evaluating_LLM_Models.py
    ```

### **Step 3: Voila!**

Navigate to the URL to test your application.
```sh
http://localhost:8501/
```
