# 🚀 Quickstart Guide

In this quick start guide, we'll demonstrate how to create a Chat with YouTube video RAG application using BeyondLLM with less than 8 lines of code. This 8 lines of code includes:

* Getting custom data source
* Retrieving documents
* Generating LLM responses
* Evaluating embeddings
* Evaluating LLM responses

## Chat with YouTube Video

### Approach-1: Using Default LLM and Embeddings

Build customised RAG in less than 5 lines of code using BeyondLLM.&#x20;

```python
from beyondllm import source,retrieve,generator
from getpass import getpass
import os
os.environ['GOOGLE_API_KEY'] = getpass("Your Google API Key:")

data = source.fit("https://www.youtube.com/watch?v=oJJyTztI_6g",dtype="youtube",chunk_size=512,chunk_overlap=50)
retriever = retrieve.auto_retriever(data,type="normal",top_k=3)
pipeline = generator.Generate(question="what tool is video mentioning about?",retriever=retriever)

print(pipeline.call())
```

### Approach-2: With Custom LLM and Embeddings

BeyondLLM support various Embeddings and LLMs that are two very important components in Retrieval Augmented Generation.&#x20;

```python
from beyondllm import source,retrieve,embeddings,llms,generator
import os
from getpass import getpass
os.environ['OPENAI_API_KEY'] = getpass("Your OpenAI API Key:")

data = source.fit("https://www.youtube.com/watch?v=oJJyTztI_6g",dtype="youtube",chunk_size=1024,chunk_overlap=0)
embed_model = embeddings.OpenAIEmbeddings()
retriever = retrieve.auto_retriever(data,embed_model,type="normal",top_k=4)
llm = llms.ChatOpenAIModel()
pipeline = generator.Generate(question="what tool is video mentioning about?",retriever=retriever,llm=llm)

print(pipeline.call()) #AI response
print(retriever.evaluate(llm=llm)) #evaluate embeddings
print(pipeline.get_rag_triad_evals()) #evaluate LLM response
```

**Output**

```markup
The tool mentioned in the context is called Jupiter, which is an AI Guru designed to simplify the learning of complex data science topics. Users can access Jupiter by logging into AI Planet, accessing any course for free, and then requesting explanations of topics from Jupiter in various styles, such as in the form of a movie plot. Jupiter aims to make AI education more accessible and interactive for everyone.

Hit_rate:1.0
MRR:1.0

Context relevancy Score: 8.0
Answer relevancy Score: 7.0
Groundness score: 7.67
```

## Core Concepts

### Load the document

The fit function from beyondllm.source module loads and processes diverse data sources, returning a List of TextNode objects, enabling integration into the RAG pipeline for question answering and information retrieval. In the code snippet below, we have a YouTube video link with the "dtype" as youtube.

```python
from beyondllm.source import fit

data = fit("https://www.youtube.com/watch?v=oJJyTztI_6g",dtype="youtube",chunk_size=1024,chunk_overlap=0)
```

### Embeddings

BeyondLLM leverages embeddings from beyondllm.embeddings to transform text into numerical representations, enabling similarity search and retrieval of relevant information. BeyondLLM provides different embedding options including Gemini, Hugging Face, OpenAI, Qdrant Fast, and Azure AI embeddings, allowing the users to select models based on preferences for efficient text representation. Here, we are using the Openai embeddings.

```python
from beyondllm.embeddings import OpenAIEmbeddings

import os
os.environ['OPENAI_API_KEY'] = "<your-api-key>"

embed_model = OpenAIEmbeddings()
```

### Auto Retriever

BeyondLLM offers various retriever types including Normal Retriever, Flag Embedding Reranker Retriever, Cross Encoder Reranker Retriever, and Hybrid Retriever, allowing efficient retrieval of relevant information based on user queries and data characteristics. In this case, we are using Normal Retriever.

```python
from beyondllm.retrieve import auto_retriever

retriever = auto_retriever(data,embed_model,type="normal",top_k=4)
```

### LLM

Large Language Models (LLMs), such as Gemini, ChatOpenAI, HuggingFaceHub, Ollama, and AzureOpenAI, are significant components within BeyondLLM, utilized in generating the responses. These models vary in architectures and capabilities, providing users with options to tailor their LLM selection based on specific requirements and preferences. In this scenario, we are using ChatOpenai LLM.

```python
from beyondllm.llms import ChatOpenAIModel
import os
os.environ['OPENAI_API_KEY'] = "<your-api-key>"

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

#returns:
#Hit_rate:1.0
#MRR:1.0
```

#### Evaluate  LLM Response

```python
print(pipeline.get_rag_triad_evals())

#returns:
#Context relevancy Score: 8.0
#Answer relevancy Score: 7.0
#Groundness score: 7.67
```
