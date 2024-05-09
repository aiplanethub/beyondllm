# 🧬 Embeddings

## What is Embedding?

Embeddings play a crucial role in BeyondLLM (Retrieval Augmented Generation) by transforming text data into numerical representations. These representations capture the semantic meaning and relationships between words and sentences, enabling efficient similarity search and retrieval of relevant information.

Embedding models can be imported from `beyondllm.embeddings` and the `.embed_text("<your-text-here>")` function can be used to get the embeddings for your text.

BeyondLLM offers several embedding models with varying characteristics and performance levels. The **default** embedding model is set as **Gemini Embedding** model. Here's a breakdown of the available options:

### **1. Hugging Face Embeddings**

This option utilizes models from the Hugging Face Hub, a vast repository of pre-trained embedding models. This will load the model and work on your data locally. The following embeddings require an additional library that can be installed with the command:

```bash
pip install llama-index-embeddings-huggingface
```

**Parameters:**

* `model_name`: Specifies the name of the Hugging Face model to use. The default is `BAAI/bge-small-en-v1.5`.

**Code Example:**

```python
from beyondllm.embeddings import HuggingFaceEmbeddings

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_embeddings = embed_model.embed_text("Huggingface models are awesome!")
```

### **2. OpenAI Embeddings**

Leverages OpenAI's embedding models, known for their quality and performance.  The following embeddings require an additional library that can be installed with the command:

```bash
pip install llama-index-embeddings-openai
```

**Parameters:**

* `api_key`: Your OpenAI API key. You can set it in the environment variable `OPENAI_API_KEY` or provide it directly.
* `model_name`: The specific OpenAI embedding model to use. The default is `text-embedding-3-small`.

**Code Example:**

```python
from beyondllm.embeddings import OpenAIEmbeddings

embed_model = OpenAIEmbeddings(api_key="your_openai_api_key", model_name="text-embedding-ada-002")
text_embeddings = embed_model.embed_text("OpenAI embeddings perform the best.")
```

### **3. Qdrant Fast Embeddings**

Employs fast and efficient embedding models optimized for Qdrant, a vector similarity search engine. (will be added in the future) It requires the following installation:

```bash
pip install llama-index-embeddings-fastembed
```

**Parameters:**

* `model_name`: The name of the Fast Embed model. The default is `BAAI/bge-small-en-v1.5`.

**Code Example:**

<pre class="language-python"><code class="lang-python">from beyondllm.embeddings import FastEmbedEmbeddings

embed_model = FastEmbedEmbeddings(model_name="your_fast_embed_model_name")
<strong>text_embeddings = embed_model.embed_text("FastEmbedding model is the default model for the retriever")
</strong></code></pre>

### **4. Azure AI Embeddings**

Integrates with Azure AI's embedding models, providing another option for high-quality text representations. The below command needs to be run to install the required library:

```bash
pip install llama-index-embeddings-azure_openai
```

**Parameters:**

* `azure_key`: Your Azure AI API key.
* `endpoint_url`: The endpoint URL for your Azure AI service.
* `api_version`: The API version to use.
* `deployment_name`: The deployment name of your embedding model in Azure AI.

**Code Example:**

```python
from beyondllm.embeddings import AzureAIEmbeddings

embed_model = AzureAIEmbeddings(
    endpoint_url="your_endpoint_url",
    azure_key="your_azure_api_key",
    api_version="your_api_version",
    deployment_name="your_deployment_name"
)
text_embedding = embed_model.embed_text("Azure embeddings models are very reliable.")
```

### 4. Gemini Embeddings

Finally, the default embedding model for our Auto Retriever: Leverages Google's powerful Gemini Text Embedding model using the default model name to be: **models/embedding-001**, it offers a robust solution for generating text representations within BeyondLLM.

#### Parameters:

* **api\_key:** Your Google API key. You can set it as an environment variable named `GOOGLE_API_KEY` or provide it directly during initialization.
* **model\_name:** The specific Gemini embedding model to use. The default is `models/embedding-001`.

#### Code Example:

```python
from beyondllm.embeddings import GeminiEmbeddings

embed_model = GeminiEmbeddings(api_key="your_google_api_key", model_name="models/embedding-001")
text_embeddings = embed_model.embed_text("Gemini embeddings offer powerful text representations.")
```

### Choosing the Right Embedding Model

Selecting the best embedding model depends on your specific requirements, such as desired accuracy, performance(Hit Rate and MRR can be calculated using our BeyondLLM!), cost, and integration preferences. Consider the following factors:

* **Accuracy:** OpenAI and Azure AI models are generally known for their high accuracy.
* **Performance:** Fast Embed models are optimized for speed and efficiency.
* **Cost:** Hugging Face Hub offers a wide range of free and open-source models, while OpenAI and Azure AI typically involve usage-based costs.
* **Integration:** Choose the option that best aligns with your existing infrastructure and workflows.

Experimenting with different models and evaluating their performance on your data can be done by BeyondLLM which will allow you to choose the best model for your use case!
