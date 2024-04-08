# Multilingual RAG Application

## Import the required libraries

```python
from enterprise_rag import source,retrieve,embeddings,llms,generator
```

## Setup API keys

```python
import os
from getpass import getpass
os.environ['OPENAI_API_KEY'] = getpass("OpenAI API Key:")
```

## Load the Source Data

Here we will use a Website as the source data. Reference: [https://www.christianitytoday.com/ct/2023/june-web-only/same-sex-attraction-not-threat-zh-hant.html](https://www.christianitytoday.com/ct/2023/june-web-only/same-sex-attraction-not-threat-zh-hant.html)

This article on Same-Sex attraction is not a threat - A Chinesse blog article. 

```python
data = source.fit(path="https://www.christianitytoday.com/ct/2023/june-web-only/same-sex-attraction-not-threat-zh-hant.html", dtype="url", chunk_size=512,chunk_overlap=0)
```

## Embedding model

We use ``intfloat/multilingual-e5-large``, a Multilingual Embedding Model from HuggingFace.

```python
embed_model = embeddings.HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
```

## Auto retriever to retrieve documents

```python
retriever = retrieve.auto_retriever(data,embed_model=embed_model,type="normal",top_k=4)
```

## Large Language Model

```python
llm = llms.ChatOpenAIModel()
```

## Define Custom System Prompt

```python
system_prompt = """ You are an Chinese AI Assistant who answers user query from the given CONTEXT \
You are honest, coherent and don't halluicnate \
If the user query is not in context, simply tell `I don't know not in context`
"""
query = "根据给定的博客，基督徒对同性恋的看法是什么"
```

## Run Generator Model

```python
pipeline = generator.Generate(question=query,system_prompt=system_prompt,retriever=retriever,llm=llm)
print(pipeline.call())
```

### Output

```bash
response:

基督教对同性恋的看法在不同派别和个人之间有所不同。保守派可能认为同性恋是不符合圣经教导的罪恶行为，
而自由派则更加包容和接纳多样性。在上文提到的博客中，作者表达了作为一个受同性吸引的基督徒对待这一议题的个人经历和观点。他们强调了身为同性吸引基督徒也可以过上充实生活，
并希望为那些建立在非血缘和性关系基础上的"属天家庭"树立榜样，认为这样的关系将会持续到永恒，而婚姻则并非如此。总的来说，这篇博客传达了对待同性恋议题时的理解和态度。
```