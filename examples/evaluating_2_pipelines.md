# Evaluating 2 Different Pipelines

## Import the required libraries

```python
from beyondllm import source,retrieve,embeddings,llms,generator
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
data = source.fit(path="build-career-in-ai.pdf", dtype="pdf", chunk_size=1024,chunk_overlap=150)
```

## Embedding model

We will use ``OpenAIEmbeddings``


```python
embed_model = embeddings.OpenAIEmbeddings()
```

## Auto retriever to retrieve documents
Since We are comparing 2 different Experiments we need to have 2 different retrievers. In this example we will use Normal and Hybrid retrievers. 
Experiment 1:
```python
retriever_normal = retrieve.auto_retriever(data,embed_model,type="normal",top_k=4)
```
Experiment 2:
```python
retriever_hybrid = retrieve.auto_retriever(data,embed_model,type="hybrid",top_k=4)
```


## Large Language Model
In this example we will be comparing 2 different LLMs, for Experiment 1 we will be using OpenAI's gpt-3.5-turbo and for Experiment 2 we will be using OpenAI's gpt-4-turbo
Experiment 1:
```python
llm_base = llms.ChatOpenAIModel()
```
Experiment 2:
```python
llm_advanced = llms.ChatOpenAIModel(model="gpt-4-0125-preview")
```


## Making a Query
Since we want to compare the responses from the two experiments, we will keep the query constant.
```python
question = 'how do I get better in AI?'
```

## Run Generator Model
Experiment 1:
```python
pipeline_normal = generator.Generate(question=question,retriever=retriever_normal,llm=llm_base)
print(pipeline_normal)
```

Experiment 2:
```python
pipeline_advanced = generator.Generate(question="How do i get better in AI?",retriever=retriever_hybrid,llm=llm_advanced)
print(pipeline_advanced)
```

## Running Evaluation
Now time to test the two pipelines to see if changes between the two made any difference. 

### Experiment 1
```python
print(pipeline_normal.call()) 
print(pipeline_normal.get_rag_triad_evals())
```


#### Output

```bash
response:
To get better in AI, there are several key steps you can take based on the provided context from the eBook:

1. **Recognize Your Strengths**: Identify what you excel at and build upon that foundation. Even understanding and explaining a portion of AI concepts to others is a step in the right direction.

2. **Continuous Learning**: AI is a rapidly evolving field, so continuous learning is essential. Deepen your technical knowledge by studying specific areas such as natural language processing, computer vision, probabilistic graphical models, or scalable software systems.

3. **Structured Learning**: While there is a wealth of information available online, enrolling in well-organized courses is often the most time-efficient way to master complex topics. This approach ensures a coherent understanding of the subject matter.

4. **Collaborate and Build a Supportive Community**: Working on projects with stakeholders and forming alliances with peers who share your goals can enhance your AI journey. Supportive mentors, peers, or a community can provide guidance and motivation.

5. **Career Growth in AI**: To succeed in AI, teamwork skills are vital. Collaborating effectively with others, improving interpersonal and communication skills, and building a strong professional network or community can propel your career forward.

By following these steps, you can enhance your skills in AI and progress on your path toward success in this dynamic and challenging field.
Executing RAG Triad Evaluations...
Context relevancy Score: 7.0, Needs Improvement
Answer relevancy Score: 8.0, Good Score
Groundness score: 6.6, Needs Improvement
```

### Experiment 2
```python
print(pipeline_advanced.call()) 
print(pipeline_advanced.get_rag_triad_evals()) 
```


#### Output
```bash
To get better in AI, follow the three-step process of career growth outlined in the context:

1. **Learning**: Master foundational skills in machine learning and deep learning through coursework. Since AI technology keeps evolving, staying up-to-date with these changes is crucial. Learning foundational skills is a career-long effort.

2. **Projects**: Work on projects, despite the challenges they present, such as difficulty in finding suitable projects, estimating timelines, and managing the highly iterative nature of AI projects. Collaboration with stakeholders who may not have expertise in AI is also part of the process.

3. **Job**: While searching for a job in AI, prepare for unique challenges due to the nascent nature of the field. Educating potential employers about your work may be necessary because many companies are still understanding which AI skills are needed.

Additionally, it is helpful to build a supportive community of friends and allies to make the journey smoother, whether you are taking your first steps in AI or have been in the field for years.
Executing RAG Triad Evaluations...
Context relevancy Score: 9.0, Good Score
Answer relevancy Score: 10.0, Good Score
Groundness score: 9.7, Good Score
```
