# 🤖 Auto Retriever

### What is Auto Retriever?

Retrievers are essential components in BeyondLLM, responsible for efficiently fetching relevant information from the knowledge base based on user queries. They utilize the generated embeddings to perform similarity search and identify the most pertinent documents or passages.&#x20;

We call this function the auto retriever, since it abstracts away all the complexity and allows you to define your retrieval type and rerankers all in one line. The `auto_retriever` function from `beyondllm.retrieve` allows you to set your retriever model.&#x20;

The `.retrieve("<your-text-here>")` function can then be used to get the List of NodeWithScore objects of your retrieval.

The auto\_retriever function in BeyondLLM allows seamless integration with vector databases, streamlining the retrieval process.

**Considerations for Vector DB in auto\_retriever:**

* **Data and Vector Database Interaction:**
  * **Optional Data:** If a vectordb instance is provided, the data argument becomes optional. This means you can retrieve information directly from the existing data within the vector database without providing additional data.
  * **Data Integration:** If both data and vectordb are provided, the new data will be added to the existing data in the vector database, creating a combined dataset for retrieval.
  * **Requirement for Hybrid Retriever:** When using the hybrid retriever type, the data argument is mandatory. This is because the hybrid retriever employs a keyword-based search component, which requires access to the raw text data.

```python
from beyondllm.retrieve import auto_retriever
from beyondllm.vectordb import ChromaVectorDb

# Initialize retriever with vector store and optional data
retriever = auto_retriever(
    data=data,
    embed_model=embed_model,
    vectordb = ChromaVectorDb(collection_name="my_collection")
)

# Perform retrieval operations
results = retriever.retrieve(query="your_user_query")
```

BeyondLLM provides several retriever types, each offering distinct approaches to information retrieval:

### **1. Normal Retriever**

This is the most basic retriever, employing vector similarity search to find the top-k most similar documents to the user query based on their embeddings.

**Parameters:**

* `data`: The dataset containing the text data (already processed and split into nodes).
* `embed_model`: The embedding model used to generate embeddings for the data.
* `top_k`: The number of top results to retrieve.

**Code Example:**

```python
from beyondllm.retrieve import auto_retriever 

# data = get data from the fit function from enterprise_rag.source 
# embed_model = get embed model from enterprise_rag.embeddings 

retriever = auto_retriever( 
    data=data, 
    embed_model=embed_model, 
    type="normal", 
    top_k=5
 ) 
 
 retrieved_nodes = retriever.retrieve("<your-query>")
```

### **2. Flag Embedding Reranker Retriever**

This retriever enhances the normal retrieval process by incorporating a "flag embedding" reranker. The reranker further refines the initial results by considering the relevance of each retrieved document to the specific query, potentially improving retrieval accuracy.

**Installation**

```bash
pip install llama-index-postprocessor-flag-embedding-reranker
pip install FlagEmbedding
```

**Parameters:**

* `data`: The dataset containing the text data (already processed and split into nodes).
* `embed_model`: The embedding model used to generate embeddings for the data.
* `top_k`: The number of top results to initially retrieve before reranking.
* `reranker`: The name of the flag embedding reranker model. The default is `BAAI/bge-reranker-large`.

**Code Example:**

```python
from beyondllm.retrieve import auto_retriever

# data = get data from the fit function from enterprise_rag.source
# embed_model = get embed model from enterprise_rag.embeddings
retriever = auto_retriever(
    data=data, 
    embed_model=embed_model, 
    type="flag-rerank", 
    top_k=5
)
retrieved_nodes = retriever.retrieve("<your-query>")
```

### **3. Cross Encoder Reranker Retriever**

Similar to the Flag Embedding Reranker, this retriever uses a cross-encoder model to rerank the initial retrieval results. Cross-encoders directly compare the query and document embeddings, often leading to more accurate relevance assessments.

**Installation**

```bash
pip install torch sentence-transformers
```

**Parameters:**

* `data`: The dataset containing the text data (already processed and split into nodes).
* `embed_model`: The embedding model used to generate embeddings for the data.
* `top_k`: The number of top results to initially retrieve before reranking.
* `reranker`: The name of the cross-encoder reranker model. The default is `cross-encoder/ms-marco-MiniLM-L-2-v2`.

**Code Example:**

```python
from beyondllm.retrieve import auto_retriever

# data = get data from enterprise_rag.source fit function
# embed_model = get embed model from enterprise_rag.embeddings
retriever = auto_retriever(
    data=data, 
    embed_model=embed_model, 
    type="cross-rerank", 
    top_k=5
)
retrieved_nodes = retriever.retrieve("<your-query>")
```

### **4. Hybrid Retriever**

This retriever combines the strengths of both vector similarity search and keyword-based search. It retrieves documents that are both semantically similar to the query and contain relevant keywords, potentially providing more comprehensive results.

**Parameters:**

* `data`: The dataset containing the text data (already processed and split into nodes).
* `embed_model`: The embedding model used to generate embeddings for the data.
* `top_k`: The number of top results to retrieve for each search method (vector and keyword) before performing OR/AND operation.&#x20;
* `mode`: Determines how results are combined. Options are `AND` (intersection of results) or `OR` (union of results). The default is `AND`.

NOTE: In case mode="OR", `top_k` nodes will be retrieved, in case of mode="AND", the number of nodes retrieved will be lesser than or equal to `top_k`

**Code Example:**

```python
from beyondllm.retrieve import auto_retriever

# data = get data from enterprise_rag.source fit function
# embed_model = get embed model from enterprise_rag.embeddings
retriever = auto_retriever(
    data=data, 
    embed_model=embed_model, 
    type="hybrid", 
    top_k=5,
    mode="OR"
)
retrieved_nodes = retriever.retrieve("<your-query>")
```

### Choosing the Right Retriever

The choice of retriever depends on your specific needs and the nature of your data:

* **Normal Retriever:** Suitable for straightforward retrieval tasks where basic semantic similarity is sufficient.
* **Reranker Retrievers:** Useful when higher accuracy is required and computational resources allow for reranking.
* **Hybrid Retriever:** Beneficial when dealing with diverse queries or when keyword relevance is important alongside semantic similarity.
