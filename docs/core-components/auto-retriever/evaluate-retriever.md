# 🔫 Evaluate retriever

### Retriever Evaluation in BeyondLLM

Evaluating the performance of your chosen retriever is crucial for ensuring the effectiveness and accuracy of your BeyondLLM application. Evaluating retrievers helps:

* **Measure Retrieval Quality:** Quantify how well the retriever identifies relevant information from the knowledge base based on user queries.
* **Compare Different Retrievers:** Assess and compare the performance of various retriever types `(Normal, Reranker, Hybrid)` to determine the best option for your specific application.
* **Optimize Retrieval Parameters:** Fine-tune parameters like `top_k` and `reranker models` to improve retrieval effectiveness.

### Evaluation Metrics

BeyondLLM offers two key metrics for retriever evaluation:

* **Hit Rate:** This metric represents the percentage of queries where the retriever successfully retrieves at least one relevant document from the knowledge base. A higher hit rate indicates better overall retrieval performance.
* **Mean Reciprocal Rank (MRR):** This metric considers the ranking of relevant documents within the retrieved results. It calculates the reciprocal of the rank of the first relevant document for each query and averages these values across all queries. A higher MRR signifies that relevant documents are ranked higher in the retrieval results.

### Evaluation Process with retriever.evaluate(llm)

The retriever.evaluate(llm) function facilitates the evaluation process by automatically generating question-answer pairs from your data using the provided Large Language Model (LLM). These QA pairs are then used to assess the retriever's performance based on the hit rate and MRR metrics.

**Here's how it works:**

1. **QA Pair Generation:** The LLM is prompted to generate questions based on the content of your knowledge base. For each piece of text (node) in your data, the LLM creates a set of questions that are likely to be answered by that specific text segment.
2. **Retrieval and Evaluation:** Each generated question is used as a query to the retriever. The retrieved documents are then compared to the expected relevant document (the one from which the question was generated). The hit rate and MRR are calculated based on whether the retriever successfully identified the correct document and its ranking within the results.

### **Important Considerations:**

* **LLM Calls:** Generating QA pairs requires multiple LLM calls, which can be time-consuming and resource-intensive, depending on the size of your knowledge base and the number of questions generated per text segment.
* **LLM Capabilities:** The quality of the generated QA pairs significantly impacts the evaluation results. Ensure your chosen LLM has adequate question generation capabilities and is aligned with the domain and content of your knowledge base.

### **Example Usage:**

```python
from beyondllm.retrieve import auto_retriever
from beyondllm.source import fit
from beyondllm.retrieve import auto_retriever
from beyondllm.llms import ChatOpenAIModel

data = fit(path="<your-doc-path-here>", dtype="<your-dtype>")
retriever = auto_retriever(data=data, type="normal", top_k=5) # takes default FastEmbedEmbeddings model

# used for generating QA pairs in evaluation
llm = ChatOpenAIModel(model="gpt-3.5-turbo",api_key = "",model_kwargs = {"max_tokens":512,"temperature":0.1})  

results = retriever.evaluate(llm)

print(f"Hit Rate: {results['hit_rate']}")
print(f"MRR: {results['mrr']}")
```
