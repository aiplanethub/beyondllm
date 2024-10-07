---
description: >-
  This section demonstrates how to evaluate a LlamaIndex pipeline using 
  BeyondLLM. We'll walk through the process step-by-step, explaining each
  component and its purpose.
---

# 🦙 LlamaIndex

## LlamaIndex  Evaluation

This section demonstrates how to evaluate a LlamaIndex pipeline using Mistral AI and BeyondLLM. We'll walk through the process step-by-step, explaining each component and its purpose.

### Setup and Imports

First, let's import the necessary libraries and set up our environment:

```python
import os
from getpass import getpass
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import chromadb
from beyondllm.utils import CONTEXT_RELEVENCE, GROUNDEDNESS, ANSWER_RELEVENCE
import re
import numpy as np
import pysbd

# Set up Hugging Face API Token
HUGGINGFACEHUB_API_TOKEN = getpass("API:")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
```

This code sets up the necessary imports and securely prompts for the Hugging Face API token.

### Document Loading and Model Configuration

Next, we'll load our documents and configure the embedding and language models:

```python
# Load documents
documents = SimpleDirectoryReader("/content/sample_data/Data").load_data()

# Configure embeddings and language model
embed_model = FastEmbedEmbedding(model_name="thenlper/gte-large")
llm = HuggingFaceInferenceAPI(
    model_name="mistralai/Mistral-7B-Instruct-v0.2", token=HUGGINGFACEHUB_API_TOKEN
)
```

Here, we load documents from a specified directory and set up our embedding model (FastEmbedEmbedding) and language model (Mistral AI via Hugging Face API).

### Vector Store and Index Setup

Now, let's set up our vector store and create an index:

```python
# Initialize Chroma Vector Store
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)
```

This code initializes a Chroma vector store, creates a collection, and builds an index from our documents using the configured embedding model.

### Utility Functions

We'll define some utility functions to help with our evaluation:

```python
def extract_number(text):
    match = re.search(r'\d+(\.\d+)?', text)
    return float(match.group()) if match else 0

def sent_tokenize(text):
    seg = pysbd.Segmenter(language="en", clean=False)
    return seg.segment(text)
```

These functions help extract numerical scores from text and tokenize sentences for evaluation.

### Evaluation Functions

Now, let's implement our evaluation functions:

```python
def get_context_relevancy(llm, query, context):
    total_score = 0
    score_count = 0
    for content in context:
        score_response = llm.complete(CONTEXT_RELEVENCE.format(question=query, context=content))
        score = float(extract_number(score_response.text))
        total_score += score
        score_count += 1
    average_score = total_score / score_count if score_count > 0 else 0
    return f"Context Relevancy Score: {round(average_score, 1)}"

def get_answer_relevancy(llm, query, response):
    score_response = llm.complete(ANSWER_RELEVENCE.format(question=query, context=response))
    return f"Answer Relevancy Score: {score_response.text}"

def get_groundedness(llm, query, context, response):
    total_score = 0
    score_count = 0
    statements = sent_tokenize(response)
    for statement in statements:
        score_response = llm.complete(GROUNDEDNESS.format(statement=statement, context=" ".join(context)))
        score = float(extract_number(score_response.text))
        total_score += score
        score_count += 1
    average_groundedness = total_score / score_count if score_count > 0 else 0
    return f"Groundedness Score: {round(average_groundedness, 1)}"
```

These functions evaluate context relevancy, answer relevancy, and groundedness of the model's responses.

### Evaluation Execution

Finally, let's execute our evaluation:

```python
# Set up query engine
query_engine = index.as_query_engine()

# Example queries
queries = [
    "what doesnt cause heart diseases",
    "what is the capital of turkey"
]

for query in queries:
    print(f"\nQuery: {query}")
    retrieved_documents = query_engine.retrieve(query)
    context = [doc.node.text for doc in retrieved_documents]
    response = query_engine.query(query)

    print(get_context_relevancy(llm, query, context))
    print(get_answer_relevancy(llm, query, response.response))
    print(get_groundedness(llm, query, context, response.response))
```

This code sets up a query engine, defines example queries, and runs the evaluation for each query.

### Sample Output

Here's a sample output of the evaluation:

```
Query: what doesnt cause heart diseases
Context Relevancy Score: 3.0
Answer Relevancy Score: 6
Groundedness Score: 6.0

Query: what is the capital of turkey
Context Relevancy Score: 0.0
Answer Relevancy Score: 0
Groundedness Score: 5.0
```

This output shows the evaluation scores for context relevancy, answer relevancy, and groundedness for each query. The scores indicate how well the model performed in retrieving relevant context, providing relevant answers, and ensuring the answers are grounded in the provided context.
