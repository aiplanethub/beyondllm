# Chat with your Powerpoint file

## Import the required libraries

```python
from enterprise_rag import source,retrieve,embeddings,llms,generator
```

## Setup API key

```python
import os
from getpass import getpass
os.environ['GOOGLE_API_KEY'] = getpass('Put the Google API Key here')
```

## Load the Source Data

Here we will use a sample powerpoint on Document Generation Using ChatGPT. You have to provide the path to your ppt file here 

```python
data = source.fit("path/to/your/powerpoint/file",dtype="ppt",chunk_size=512,chunk_overlap=51)
```

## Embedding model

We have the default Embedding Model which is FastEmbeddings in this case.

## Auto retriever to retrieve documents

```python
retriever = retrieve.auto_retriever(data,type="normal",top_k=3)
```

## Run Generator Model

```python
pipeline = generator.Generate(question="what is this powerpoint presentation about?",retriever=retriever)
print(pipeline.call())
```

### Output

```bash
Sample response:

The presentation focuses on exploring the depths of document generation using GPT-3.5. It entails a detailed walkthrough of the methodologies employed, shedding light on the current state, and presenting avenues for future advancements.
```
