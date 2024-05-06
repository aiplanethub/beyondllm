# 📚 Re-ranker Retrievers

## Enhancing Retrieval Accuracy

While basic vector similarity search is a valuable tool for information retrieval, it may not always perfectly capture the nuanced relevance of documents to specific user queries. This is where reranking techniques come into play, further refining the initial retrieval results to prioritize the most pertinent information.

## Importance of Reranking

Reranking offers several advantages:

* **Improved Relevance:** Reranking models can better assess the semantic relationship between the query and retrieved documents, leading to more accurate identification of truly relevant information.
* **Enhanced User Experience:** By presenting the most pertinent results first, reranking improves the user experience and reduces the time spent sifting through potentially less relevant documents.

## Automating Reranking with BeyondLLM

BeyondLLM simplifies the implementation of reranking techniques by providing built-in support for two popular methods. We allow you to implement re-ranking with a single parameter in the retriever declaration, just set the `type` parameter to one of the below and set the `reranker` parameter to the reranking model you want to use:

* **Flag Embedding Reranker:** This method utilizes a specialized model trained to assess the relevance of documents to queries based on their flag embeddings, which capture additional information beyond basic semantic similarity. Default model is: `BAAI/bge-reranker-large`
* **Cross-Encoder Reranker:** This method employs a cross-encoder model, which directly compares the query and document embeddings to determine their relevance. Cross-encoders often achieve higher accuracy but may require more computational resources. Default model is : `cross-encoder/ms-marco-MiniLM-L-2-v2`

_**Since we are using the models to re-rank, it takes split of seconds to load the model for first time.**_&#x20;

## Code Example: Reranking and Evaluation

This example demonstrates the use of a cross-encoder reranker retriever, leveraging `LlamaParse` for data loading and incorporating evaluation steps for both retriever performance and LLM response quality.

### 1. Load and Process Data with LlamaParse

The fit function processes and prepares your data for indexing and retrieval using LlamaParse. LlamaParse extracts structured information from documents, including headings and other formatting elements, and converts it into markdown format. This preserves valuable metadata about the document structure, which can be beneficial for retrieval and generation tasks.

```python
from beyondllm.source import fit

data = fit(path="your_data_file.pdf", dtype="llama-parse", chunk_size=512, chunk_overlap=100, llama_parse_key="your_llama_parse_api_key")
```

### 2. Load Embedding Model

The chosen embedding model generates vector representations of the text data extracted by LlamaParse. These embeddings capture the semantic meaning of the text and enable efficient similarity search during retrieval.

```python
from beyondllm.embeddings import HuggingFaceEmbeddings

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### 3. Initialize Retriever with Cross-Encoder Reranking

The auto\_retriever function creates a retriever with the specified type ("cross-rerank" in this case) and parameters. The retriever utilizes the embeddings generated in step 2 to perform similarity search and retrieve relevant documents. Additionally, the cross-encoder reranker refines the initial results by directly comparing query and document embeddings for improved accuracy.

```python
from beyondllm.retrieve import auto_retriever

retriever = auto_retriever(
    data=data,
    embed_model=embed_model,
    type="cross-rerank",
    top_k=5,
    reranker="cross-encoder/ms-marco-MiniLM-L-2-v2",
)
```

### 4. Load LLM for Evaluation and Generation

The LLM serves two purposes:

* **Evaluation:** It generates question-answer pairs from the knowledge base to assess the retriever's performance.
* **Generation:** It will be used later to generate responses to user queries based on the retrieved information.

```python
from beyondllm.llms import ChatOpenAIModel

llm = ChatOpenAIModel(api_key="your_openai_api_key")
```

### 5. Evaluate Retriever Performance

The evaluate function measures the retriever's effectiveness using the generated QA pairs. It calculates the hit rate (percentage of queries where a relevant document is retrieved) and MRR (mean reciprocal rank of the first relevant document) to quantify retrieval accuracy.

```python
results = retriever.evaluate(llm)

print(f"Reranker Hit Rate and MRR: {results}")
```

### 6. Generate Response and Evaluate LLM Output

This step simulates a user query and generates a response using the BeyondLLM pipeline. The Generate class combines the retriever and LLM to fetch relevant information and formulate an answer. Additionally, the RAG Triad evaluations assess the quality of the LLM's response.

```python
from beyondllm.generate import Generate

pipeline = Generate(question="<user-question-here>", retriever=retriever, llm=llm)
print(pipeline.call())  # AI response

print(pipeline.get_rag_triad_evals())  # Evaluate LLM response quality
```

### **Explanation of Evaluation Outputs:**

* **Retriever Evaluation:** The hit rate and MRR provide insights into the retriever's ability to locate relevant information.
* **RAG Triad Evaluations:**
  * **Context Relevancy:** Measures how well the retrieved information relates to the user query.
  * **Answer Relevancy:** Assesses the relevance of the generated response to the user query.
  * **Groundedness:** Evaluates whether the generated response is supported by the retrieved information and avoids hallucination.

> **Remember:** Experiment with different reranker models and retrieval parameters to optimize your Enterprise RAG application for your specific use case and data characteristics.
