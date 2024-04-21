## BeyondLLM: Core Concepts

### Load the document

The fit function from beyondllm.source module loads and processes diverse data sources, returning a List of TextNode objects, enabling integration into the RAG pipeline for question answering and information retrieval. In the code snippet below, we have a YouTube video link with the "dtype" as youtube.

```python
from beyondllm.source import fit

data = fit("https://www.youtube.com/watch?v=oJJyTztI_6g",dtype="youtube",chunk_size=1024,chunk_overlap=0)
```

### Embeddings

BeyondLLM leverages embeddings from beyondllm.embeddings to transform text into numerical representations, enabling similarity search and retrieval of relevant information. BeyondLLM provides different embedding options including Gemini, Hugging Face, OpenAI, Qdrant Fast, and Azure AI embeddings, allowing the users to select models based on preferences for efficient text representation. Here, we are using the Gemmini embeddings.

```python
from beyondllm.embeddings import GeminiEmbeddings
from getpass import getpass

os.environ['GOOGLE_API_KEY'] = getpass("Enter your Google API Key")

embed_model = GeminiEmbeddings()
```

### Auto Retriever

BeyondLLM offers various retriever types including Normal Retriever, Flag Embedding Reranker Retriever, Cross Encoder Reranker Retriever, and Hybrid Retriever, allowing efficient retrieval of relevant information based on user queries and data characteristics. In this case, we are using Normal Retriever.

```python
from beyondllm.embeddings import auto_retriever

retriever = auto_retriever(data,embed_model,type="normal",top_k=4)
```

### Large Language Model (LLM)

Large Language Models (LLMs), such as Gemini, ChatOpenAI, HuggingFaceHub, Ollama, and AzureOpenAI, are significant components within BeyondLLM, utilized in generating the responses. These models vary in architectures and capabilities, providing users with options to tailor their LLM selection based on specific requirements and preferences. In this scenario, we are using ChatOpenai LLM.

```python
from beyondllm.llms import ChatOpenAIModel

llm = ChatOpenAIModel()
```

### Generator

The generator function in BeyondLLM is the component that generates responses by leveraging retriever and LLM, enabling pipeline evaluation and response generation based on user queries and system prompts.

```python
from beyondllm.generator import Generate

query = "what tool is video mentioning about?"
pipeline = Generate(question = query,retriever = retriever,llm = llm)
print(pipeline.call())
```

### Evaluation

BeyondLLM's evaluation benchmarks, including Context Relevance, Answer Relevance, Groundedness, and Ground Truth, quantify the pipeline's performance in sourcing relevant data, generating appropriate responses, ensuring factual grounding, and aligning with predefined correct answers, respectively. Additionally, the RAG Triad method computes all three key evaluation metrics simultaneously.

#### Evaluate Embeddings

```python
print(retriever.evaluate(llm=llm))
```

```bash
#returns:
#Hit_rate:1.0
#MRR:1.0
```

#### Evaluate  LLM Response

```python
print(pipeline.get_rag_triad_evals())
```

```bash
#returns:
#Context relevancy Score: 8.0
#Answer relevancy Score: 7.0
#Groundness score: 7.67
```