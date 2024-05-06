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

## Choosing the Right Vector Database

The selection of the most suitable vector database depends on several factors:

* **Scale and Performance:** Consider the expected size of your embedding data and the required query speed.&#x20;
* **Persistence:** Determine whether you need to persist the embedding data or if an in-memory solution is sufficient.
* **Features:** Evaluate the need for advanced features like filtering, indexing, and scalability.&#x20;
