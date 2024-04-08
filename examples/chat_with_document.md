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

## Deploy Inference: Streamlit implementation
```python
import os
import streamlit as st
from enterprise_rag import source, retrieve, embeddings, llms, generator
from getpass import getpass


st.title("Chat with document")

st.text("Enter API Key")

api_key = st.text_input("API Key:", type="password")
os.environ['OPENAI_API_KEY'] = api_key

if api_key:
    st.success("API Key entered successfully!")


    uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')


    question = st.text_input("Enter your question")

    if uploaded_file is not None and question:
        
        save_path = "./uploaded_files"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        data = source.fit(file_path, dtype="pdf", chunk_size=1024, chunk_overlap=0)
        embed_model = embeddings.OpenAIEmbeddings()
        retriever = retrieve.auto_retriever(data, embed_model, type="normal", top_k=4)
        llm = llms.ChatOpenAIModel()
        pipeline = generator.Generate(question=question, retriever=retriever, llm=llm)
        response = pipeline.call()
        
        st.write(response)


st.caption("Upload a PDF document and enter a question to query information from the document.")

```