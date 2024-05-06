# 🔀 Hybrid Retrievers

## Enhancing Retrieval Accuracy

This retriever combines the strengths of vector similarity search and keyword-based search. By seamlessly blending these approaches, it retrieves documents that not only align semantically with the query but also encompass relevant keywords. The result is a more holistic and comprehensive set of results, enhancing the overall effectiveness of information retrieval.

## Code Example: Hybrid Retrievers

This example demonstrates the use of a Hybrid retriever, using evaluation steps for both retriever performance and LLM response quality.

### 1. Load and Process the Data

The fit function processes and prepares your data for indexing and retrieval. It offers a unified interface for loading and processing data regardless of the source type. Here we are using a pdf file for retrieval purposes.

```python
# fit the data from the pdf file
from beyondllm.source import fit

data = fit(path="path/to/your/pdf/file.pdf", dtype="pdf", chunk_size=512, chunk_overlap=100)
```

### 2. Load Embedding Model

The chosen embedding model generates vector representations of the text data extracted by the fit function. These embeddings capture the semantic meaning of the text and enable efficient similarity search during retrieval.&#x20;

Here we are using `all-MiniLM-L6-v2` model from the HuggingFace hub.

```python
# Load the embedding model from Hugging Face Hub
from beyondllm.embeddings import HuggingFaceEmbeddings

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### 3. Initialize Retriever with Hybrid Search

The Auto Retriever in BeyondLLM simplifies information retrieval by abstracting complexity, enabling easy configuration of retrieval types and re-rankers. With a single line, it efficiently fetches relevant documents or passages based on user queries, utilizing embeddings for similarity search.&#x20;

```python
# Initialize Retriever with Hybrid search
from beyondllm.retrieve import auto_retriever

retriever = auto_retriever(
    data=data, 
    embed_model=embed_model, 
    type="hybrid", 
    top_k=5,
    mode="OR"
)
```

### 4. Load LLM for Evaluation and Generation

The LLM serves two purposes:

* **Evaluation:** It generates question-answer pairs from the knowledge base to assess the retriever's performance.
* **Generation:** It will be used later to generate responses to user queries based on the retrieved information.

Here we are using the `zephyr-7b-beta` model from the HuggingFace hub.

```python
# Load the LLM model from HuggingFace Hub
from beyondllm.llms import HuggingFaceHubModel

llm = HuggingFaceHubModel(model="HuggingFaceh4/zephyr-7b-beta", token="your_huggingfacehub_token", model_kwargs={"max_new_tokens":512,"temperature":0.1})
```

### 5. Evaluate Retriever Performance

The evaluate function measures the retriever's effectiveness using the generated Question-Answer pairs. It calculates the hit rate (percentage of queries where a relevant document is retrieved) and MRR (mean reciprocal rank of the first relevant document) to quantify retrieval accuracy.

```python
# Evaluate the LLM model
results = retriever.evaluate(llm)

print(f"Reranker Hit Rate and MRR: {results}")
```

### 6. Generate Response and Evaluate LLM Output

This step simulates a user query and generates a response using the BeyondLLM pipeline. The Generate class combines the retriever and LLM to fetch relevant information and formulate an answer. Additionally, the RAG Triad evaluations assess the quality of the LLM's response.

```python
# Generate text using the LLM model
from beyondllm.generator import Generate

pipeline = Generate(question="what is the pdf mentioning about?", retriever=retriever, llm=llm)
print(pipeline.call())  # AI response

print(pipeline.get_rag_triad_evals())  # Evaluate LLM response quality
```

## Explanation of Evaluation Outputs:

* **Retriever Evaluation:** The hit rate and MRR provide insights into the retriever's ability to locate relevant information.
* **RAG Triad Evaluations:**
  * **Context Relevancy:** Measures how well the retrieved information relates to the user query.
  * **Answer Relevancy:** Assesses the relevance of the generated response to the user query.
  * **Groundedness:** Evaluates whether the generated response is supported by the retrieved information and avoids hallucination.

{% hint style="info" %}
**Remember:** Experiment with different re-ranker models and retrieval parameters to optimize your BeyondLLM application for your specific use case and data characteristics.
{% endhint %}
