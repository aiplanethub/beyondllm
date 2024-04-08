# Chat with Document RAG Application

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

Here we will use a short book by Andrew Ng called 'Build a career in AI.
```python
data = source.fit(path="build-career-in-ai.pdf", dtype="pdf", chunk_size=512,chunk_overlap=0)
```

## Embedding model

We will use ``OpenAIEmbeddings``


```python
embed_model = embeddings.OpenAIEmbeddings()
```

## Auto retriever to retrieve documents

```python
retriever = retrieve.auto_retriever(data,embed_model=embed_model,type="normal",top_k=4)
```

## Large Language Model

```python
llm = llms.ChatOpenAIModel()
```

## Making a Query
```python
question = 'how to excel in the field of AI?'
```

## Run Generator Model

```python
pipeline = generator.Generate(question=question, retriever=retriever, llm=llm)
print(pipeline.call())
```

### Output

```bash
response:

To excel in the field of AI, it is essential to focus on several key areas:

Foundational Skills: Develop a strong understanding of foundational machine learning concepts such as linear regression, neural networks, decision trees, and clustering. Additionally, grasp core concepts like bias/variance, cost functions, regularization, and optimization algorithms.

Deep Learning: Gain knowledge of neural networks, convolutional networks, sequence models, and transformers. Deep learning has become integral to AI, and understanding these concepts is crucial for excelling in the field.

Software Development: Enhance your skills in software development, including programming fundamentals, data structures, algorithms, and software design. Proficiency in programming languages like Python and libraries like TensorFlow or PyTorch is beneficial.

Mathematics: Develop a strong foundation in math relevant to machine learning, including linear algebra, probability, statistics, and calculus. Exploratory data analysis (EDA) is also an important skill to master for driving progress in AI projects.

Continuous Learning: AI is a rapidly evolving field, so lifelong learning is essential. Stay updated with the latest technologies and research papers. Engage in continuous learning to deepen your technical knowledge and stay ahead in the field.

Community Building: Build a supportive community of like-minded individuals in the AI field. Interact with peers, collaborate on projects, share knowledge, and seek advice. Networking and engaging with others can help propel your career forward and provide new opportunities for growth.

By focusing on these areas, continuously learning, and building a strong network within the AI community, you can position yourself for success and excel in the field of AI.
```