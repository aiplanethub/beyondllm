# 💼 Vector Store

## **What is a Vector Database?**

In the context of BeyondLLM (Retrieval Augmented Generation), a vector database plays a crucial role in efficiently storing and managing vector embeddings generated from your text data. These embeddings capture the semantic meaning and relationships within the text, enabling rapid retrieval of relevant information based on user queries.&#x20;

Vector databases are optimized for similarity search, making them essential for effective RAG applications. Available Vector Databases are:&#x20;

### 1. Chroma

BeyondLLM currently integrates with Chroma, a powerful and purpose-built vector database designed for high-performance similarity search and efficient management of vector embeddings.

**Parameters for ChromaVectorDb:**

* `collection_name` (required): Specifies the name of the collection within Chroma to store your embeddings. This helps organize and manage different sets of embeddings within the database.
* `persist_directory` (optional): The directory path to persist the Chroma database on disk. If not provided or set as an empty string (""), the Chroma instance will be ephemeral and created in memory, meaning the data will not be saved after the program ends.

**Code Example:**

<pre class="language-python"><code class="lang-python"><strong>from beyondllm.vectordb import ChromaVectorDb
</strong>
# Persistent Chroma instance with data stored on disk, else don't pass persist_directory
vectordb = ChromaVectorDb(collection_name="my_persistent_collection", persist_directory="./db/chroma/")
</code></pre>

### 2. Pinecone

Pinecone is a fully managed vector database service designed to provide high performance and scalability for similarity search applications. It offers a robust and user-friendly platform for storing, indexing, and querying vector embeddings, making it an excellent choice for BeyondLLM's Retrieval Augmented Generation (RAG) capabilities.

**Parameters:**

* `api_key` (required): Your Pinecone API key for accessing the service.
* `index_name` (required): The name of the index within Pinecone where your embeddings will be stored.
* `create` (optional): Set to True to create a new index if it doesn't exist. Defaults to False, assuming the index already exists.
* `embedding_dim` (required if create=True): The dimensionality of the embedding vectors. This is essential when creating a new index. 768 is the dimension of our default embeddings. (1536 is the size of OpenAI's Default embeddings)
* `metric` (required if create=True): The distance metric used for similarity search. Common options include "cosine" and "euclidean".
* `spec` (optional): The deployment specification. Options are "serverless" (default) or "pod-based".
* `cloud (`required for serverless`):` The cloud provider for your serverless Pinecone index.
* `region` (required for serverless): The region for your serverless Pinecone index.
* `pod_type`  (required for pod-based): The pod type for your dedicated Pinecone index.
* `replicas` (required for pod-based): The number of replicas for your dedicated Pinecone index.

**Code Example:**

To simply use an existing Pinecone index, all you need to specify is the `api_key` and the `index_name` parameters. You can pass your data which will be converted into vectors based on your embedding model and these vectors will be upserted in BeyondLLM's `auto_retriever` method.

As mentioned in the parameters, setting the `create` parameter as True allows you to create a new index that doesn't exist, this can be done by setting the spec based on your index type as shown below:

```python
from beyondllm.vectordb import PineconeVectorDb

# Connect to existing Pinecone index
vectordb_existing = PineconeVectorDb(api_key="your_api_key", index_name="your_index_name")

# Create a new serverless Pinecone index
vectordb_new_serverless = PineconeVectorDb(
    create=True,
    api_key="your_api_key",
    index_name="your_new_index_name",
    embedding_dim=768,
    metric="cosine",
    cloud="aws",
    region="us-east-1",
)

# Create a new pod-based Pinecone index: NOT AVAILABLE IN FREE TIER
vectordb_new_pod = PineconeVectorDb(
    create=True,
    api_key="your_api_key",
    index_name="your_new_index_name",
    embedding_dim=768,
    metric="cosine",
    spec="pod-based",
    pod_type="p1",
    replicas=1,
)
```

**Integrating with BeyondLLM Retrievers:**

VectorDB instances can be used with the auto\_retriever functionality provided by BeyondLLM, by simply passing instance within the auto\_retriever function to enable efficient retrieval from your Vector Store index:

```python
from beyondllm.retrieve import auto_retriever

# Initialize your vector store
vector_store = <your-vector-store-instance-here>
retriever = auto_retriever(data=data, embed_model=embed_model, type="normal", top_k=5, vectordb=vector_store)

# Perform retrieval operations
results = retriever.retrieve(query="your_user_query")
```

## Choosing the Right Vector Database

The selection of the most suitable vector database depends on several factors:

* **Scale and Performance:** Consider the expected size of your embedding data and the required query speed.&#x20;
* **Persistence:** Determine whether you need to persist the embedding data or if an in-memory solution is sufficient.
* **Features:** Evaluate the need for advanced features like filtering, indexing, and scalability.&#x20;
