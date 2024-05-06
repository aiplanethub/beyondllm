# 🔋 Generator

## What is a Generator?

The `generator` is a core component designed to generate responses. Besides generating responses you can evaluate your pipeline as well from within the generator. Generator function utilizes the retriever and llm to generate a response. It puts everything together to answer the user query.&#x20;

#### Parameters

* **User query** : The question from the user.&#x20;
* **System Prompt** Optional\[str] **:** The system prompt that directs the responses of llm.&#x20;
* **Retriever :** The retriever which will fetch relevant information from the knowledge base based on the user query.&#x20;
* **LLM** \[default: Gemini model] : The Language model to generate the response based on the information fetched by the retriever.

#### Code Snippet&#x20;

```python
from beyondllm import generator

user_prompt = "......"
# using default LLM
pipeline = generator.Generate(question=user_prompt,retriever=retriever)


from beyondllm.llms import OllamaModel
llm = Ollama(model="llama2")
system_prompt = "You are an AI assistant...."
pipeline = generator.Generate(
                 question=user_prompt
                 system_prompt = system_prompt
                 llm = llm,
                 retriever=retriever
)
```

#### Call&#x20;

Once the pipeline is setup we use the call function to return the generated response from LLM that acts as Generator in RAG.&#x20;

```python
print(pipeline.call())
```

### Evaluation

Evaluation is an integral part of BeyondLLM as it circles out the pain points in the pipeline. Generator lets you evaluate the pipeline on list of important benchmarks. For more information kindly refer to : [evaluation.md](evaluation.md "mention")&#x20;



