# 🤖 Customer Service Bot

## Customer Service ChatBot

### Import the required libraries

```py
from beyondllm import source,retrieve,embeddings,llms,generator
```

### Setup API keys

```py
import os
from getpass import getpass
os.environ['OPENAI_API_KEY'] = getpass("OpenAI API Key:")
```

### Load the Source Data

Here we will use a Website as the source data. Reference: [https://www.lacworldwide.com.my/en/protein-and-fitness\_whey-protein/optimum-nutrition/gold-standard-100-whey-double-rich-chocolate-06100030.html?catId=protein-and-fitness](https://www.lacworldwide.com.my/en/protein-and-fitness\_whey-protein/optimum-nutrition/gold-standard-100-whey-double-rich-chocolate-06100030.html?catId=protein-and-fitness)

The goal is to have a customer service chatbot that can answer to the query based on the product data given.

```py
data = source.fit(path="https://www.lacworldwide.com.my/en/protein-and-fitness_whey-protein/optimum-nutrition/gold-standard-100-whey-double-rich-chocolate-06100030.html?catId=protein-and-fitness", dtype="url", chunk_size=512,chunk_overlap=0)
```

### Embedding model

We use `OpenAIEmbeddings`, an embedding model from OpenAI.

```py
embed_model = embeddings.OpenAIEmbeddings()
```

### Auto retriever to retrieve documents

```py
retriever = retrieve.auto_retriever(data,embed_model=embed_model,type="normal",top_k=4)
```

### Large Language Model

```py
llm = llms.ChatOpenAIModel()
```

### Define Custom System Prompt

Define the system prompt, that instructs the model to behave as a customer bot

```python
system_prompt = """ You are a Customer support Assistant who answers user query from the given CONTEXT, sound like a customer service\
You are honest, coherent and don't halluicnate \
If the user query is not in context, simply tell `We are sorry, we don't have information on this` \
"""
query = "What is the price of Gold Standard 100 Whey Double Rich Chocolate?"
```

### Run Generator Model

```python
pipeline = generator.Generate(question=query,system_prompt=system_prompt,retriever=retriever,llm=llm)
print(pipeline.call())
```

#### Output

```bash
The price of Gold Standard 100% Whey Double Rich Chocolate is RM295.00 for a 5 lb container. This is the member price, which allows you to save 32%. The usual price is RM436.90. If you're interested, you can add it to your cart by clicking on the "Add to Cart" button. Let me know if you need any further assistance!
```
