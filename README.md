<div align="center">
<h1 align="center">BeyondLLM</h1>
<h2 align="center">Build - Rapid Experiment - Evaluate - Repeat</h2>

![Thumbnails](https://github.com/aiplanethub/beyondllm/assets/132284203/489bb644-f87e-4477-a639-a63552f84cd7)

<a href="https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-3776AB.svg?style=flat&logo=python&logoColor=white"><img src="https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-3776AB.svg?style=flat&logo=python&logoColor=white" alt="Python Versions"></a>
<a href="https://discord.gg/4aWV7He2QU"><img src="https://dcbadge.vercel.app/api/server/4aWV7He2QU?style=flat" alt="Discord" /></a>
<a href="https://twitter.com/aiplanethub"><img src="https://img.shields.io/twitter/follow/aiplanethub" alt="Twitter" /></a>
<a href="https://colab.research.google.com/drive/1S1UL2uCahHkfJsurRA3f7dcR6IHjg-IM?usp=sharing" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Try an example"/></a>

[![Contributors](https://img.shields.io/github/contributors/aiplanethub/beyondllm.svg?style=flat-square)](https://github.com/aiplanethub/beyondllm/graphs/contributors)
[![Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-contributor%20covenant-green.svg?style=flat-square)](CODE_OF_CONDUCT.md)

<p>Beyond LLM offers an all-in-one toolkit for experimentation, evaluation, and deployment of Retrieval-Augmented Generation (RAG) systems, simplifying the process with automated integration, customizable evaluation metrics, and support for various Large Language Models (LLMs) tailored to specific needs, ultimately aiming to reduce LLM hallucination risks and enhance reliability.</p>
<i><a href="https://discord.gg/4aWV7He2QU">👉 Join our Discord community!</a></i>
</div>

Try out a quick demo on Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1S1UL2uCahHkfJsurRA3f7dcR6IHjg-IM?usp=sharing)

## Quick install

```bash
pip install beyondllm
```

## Quickstart Guide- Chat with YouTube Video

In this quick start guide, we'll demonstrate how to create a Chat with YouTube video RAG application using Beyond LLM with less than 8 lines of code. This 8 lines of code includes:
* Getting custom data source
* Retrieving documents
* Generating LLM responses
* Evaluating embeddings
* Evaluating LLM responses

### Approach-1: Using Default LLM and Embeddings

Build customised RAG in less than ``5 lines of code`` using Beyond LLM. 

```python
from beyondllm import source,retrieve,generator
import os
os.environ['GOOGLE_API_KEY'] = "Your Google API Key:"

data = source.fit("https://www.youtube.com/watch?v=oJJyTztI_6g",dtype="youtube",chunk_size=512,chunk_overlap=50)
retriever = retrieve.auto_retriever(data,type="normal",top_k=3)
pipeline = generator.Generate(question="what tool is video mentioning about?",retriever=retriever)

print(pipeline.call())
```

### Approach-2: With Custom LLM and Embeddings

Beyond LLM support various Embeddings and LLMs that are two very important components in Retrieval Augmented Generation. 

```python
from beyondllm import source,retrieve,embeddings,llms,generator
import os
from getpass import getpass
os.environ['OPENAI_API_KEY'] = getpass("Your OpenAI API Key:")

data = source.fit("https://www.youtube.com/watch?v=oJJyTztI_6g",dtype="youtube",chunk_size=1024,chunk_overlap=0)
embed_model = embeddings.OpenAIEmbeddings()
retriever = retrieve.auto_retriever(data,embed_model,type="normal",top_k=4)
llm = llms.ChatOpenAIModel()
pipeline = generator.Generate(question="what tool is video mentioning about?",retriever=retriever,llm=llm)

print(pipeline.call()) #AI response
print(retriever.evaluate(llm=llm)) #evaluate embeddings
print(pipeline.get_rag_triad_evals()) #evaluate LLM response
```

##### Output

```bash
The tool mentioned in the context is called Jupiter, which is an AI Guru designed to simplify the learning of complex data science topics. Users can access Jupiter by logging into AI Planet, accessing any course for free, and then requesting explanations of topics from Jupiter in various styles, such as in the form of a movie plot. Jupiter aims to make AI education more accessible and interactive for everyone.

Hit_rate:1.0
MRR:1.0

Context relevancy Score: 8.0
Answer relevancy Score: 7.0
Groundness score: 7.666666666666667
```

## Observability

``Observability`` helps to keep track of the closed source models on the latency and the cost monitor tracking. BeyondLLM provides ``Observer`` that currently monitors the OpenAI LLM model performance. 

```python
from beyondllm import source,retrieve,generator, llms, embeddings
from beyondllm.observe import Observer
import os

os.environ['OPENAI_API_KEY'] = 'sk-****'

Observe = Observer()
Observe.run()

llm=llms.ChatOpenAIModel()
embed_model = embeddings.OpenAIEmbeddings()

data = source.fit("https://www.youtube.com/watch?v=oJJyTztI_6g",dtype="youtube",chunk_size=512,chunk_overlap=50)
retriever = retrieve.auto_retriever(data,embed_model,type="normal",top_k=4)

pipeline = generator.Generate(question="what tool is video mentioning about?",retriever=retriever, llm=llm)
pipeline = generator.Generate(question="What is the tool used for?",retriever=retriever, llm=llm)
pipeline = generator.Generate(question="How can i use the tool for my own use?",retriever=retriever, llm=llm)
```

## Documentation

See the [beyondllm.aiplanet.com](https://beyondllm.aiplanet.com/) for complete documentation.


## Contribution guidelines

Beyond LLM thrives in the rapidly evolving landscape of open-source projects. We wholeheartedly welcome contributions in various capacities, be it through innovative features, enhanced infrastructure, or refined documentation.

See [Contributing guide](https://github.com/aiplanethub/beyondllm/blob/main/CONTRIBUTING.md) for more information on contributing to the BeyondLLM library. 

## Acknowledgements

* [HuggingFace](https://github.com/huggingface)
* [LlamaIndex](https://github.com/jerryjliu/llama_index)
* [OpenAI](https://github.com/openai)
* [Google Gemini](https://ai.google.dev/)
  
and the entire OpenSource community.

## License

The contents of this repository are licensed under the [Apache License, version 2.0](https://github.com/aiplanethub/beyondllm/blob/main/LICENSE).

## Get in Touch

You can schedule a 1:1 meeting with our Team to get started with GenAI Stack, OpenAGI, AI Planet Open Source LLMs(Buddhi, effi and Panda Coder) and Beyond LLM. Schedule the call here: [https://calendly.com/jaintarun](https://calendly.com/jaintarun)
