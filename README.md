<div align="center">
<h1 align="center">Enterprise-RAG</h1>
<h2 align="center">Build - Quick Experiment - Evaluate - Repeat</h2>

<a href="https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-3776AB.svg?style=flat&logo=python&logoColor=white"><img src="https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-3776AB.svg?style=flat&logo=python&logoColor=white" alt="Python Versions"></a>
<a href="https://discord.gg/4aWV7He2QU"><img src="https://dcbadge.vercel.app/api/server/4aWV7He2QU?style=flat" alt="Discord" /></a>
<a href="https://twitter.com/aiplanethub"><img src="https://img.shields.io/twitter/follow/aiplanethub" alt="Twitter" /></a>
<!-- <a href="https://colab.research.google.com/drive/1y6_0MoNWjS9wugv0askP1Jb7zrY_sQT-?usp=sharing"><img src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Colab" /></a> -->

<p>Enterprise RAG offers an all-in-one toolkit for experimentation, evaluation, and deployment of Retrieval-Augmented Generation (RAG) systems, simplifying the process with automated integration, customizable evaluation metrics, and support for various Large Language Models (LLMs) tailored to specific needs, ultimately aiming to reduce LLM hallucination risks and enhance reliability.</p>
<i><a href="https://discord.gg/4aWV7He2QU">👉 Join our Discord community!</a></i>
</div>

Try out a quick demo on Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dJZF5113e5XQsm6GxuW3ShYCBZcUs3-_?usp=sharing)

## Quick install

To install Enterprise RAG i.e., a private repo, we can use Access Token of GitHub. 

```bash
git clone https://<UPDATE-WITH-YOUR-TOKEN>@github.com/aiplanethub/enterprise-rag.git
cd enterprise-rag
```

Create virtual environment

```bash
python3 -m venv env 
source env/bin/activate

or

virtualenv env
source env/bin/activate
```

Install all the packages within the virtual environment. 

```bash
pip install -r requirements.txt
```

Install on Google Colab

```bash
!git clone https://<UPDATE-WITH-YOUR-TOKEN>@github.com/aiplanethub/enterprise-rag.git

!pip install -r requirements.txt

%cd /content/enterprise_rag/

!ls
```

## Quickstart Guide- Chat with YouTube Video

In this quick start guide, we'll demonstrate how to create a Chat with YouTube video RAG application using Enterprise RAG with less than 8 lines of code. This 8 lines of code includes:
* Getting custom data source
* Retrieving documents
* Generating LLM responses
* Evaluating embeddings
* Evaluating LLM responses

### Approach-1: Using Default LLM and Embeddings

Build customised RAG in less than ``5 lines of code`` using Enterprise RAG. 

```python
from enterprise_rag import source,retrieve,generator
import os
os.environ['GOOGLE_API_KEY'] = "Your Google API Key:"

data = source.fit("https://www.youtube.com/watch?v=oJJyTztI_6g",dtype="youtube",chunk_size=512,chunk_overlap=50)
retriever = retrieve.auto_retriever(data,type="normal",top_k=3)
pipeline = generator.Generate(question="what tool is video mentioning about?",retriever=retriever)

print(pipeline.call())
```

### Approach-2: With Custom LLM and Embeddings

Enterprise RAG support various Embeddings and LLMs that are two very important components in Retrieval Augmented Generation. 

```python
from enterprise_rag import source,retrieve,embeddings,llms,generator
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

## Get in Touch

You can schedule a 1:1 meeting with our DevRel & Community Team to get started with AI Planet Open Source LLMs(effi and Panda Coder) and Enterprise RAG. Schedule the call here: [https://calendly.com/jaintarun](https://calendly.com/jaintarun)

## Contribution guidelines

Enterprise RAG thrives in the rapidly evolving landscape of open-source projects. We wholeheartedly welcome contributions in various capacities, be it through innovative features, enhanced infrastructure, or refined documentation.

## Acknowledgements

* [HuggingFace](https://github.com/huggingface)
* [LlamaIndex](https://github.com/jerryjliu/llama_index)
* [OpenAI](https://github.com/openai)
* [Google Gemini](https://ai.google.dev/)
  
and the entire OpenSource community.
