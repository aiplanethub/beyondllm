# 🦜️🔗 Langchain

**Introduction**

This section delves into the seamless integration of BeyondLLM with LangChain, a powerful toolkit for constructing and evaluating intelligent systems. By harnessing the combined capabilities of these tools, we'll demonstrate the creation of a robust document retrieval and question-answering (QA) system empowered by Retrieval-Augmented Generation (RAG).

**Installation**

The following code snippet installs the essential Python packages required for this integration:

```
!pip install langchain sentence-transformers chromadb llama-cpp-python langchain_community pypdf langchain-groq
!pip install beyondllm
!pip install faiss-cpu
```

**Importing Necessary Libraries**

Next, we import the necessary libraries to work with LangChain, document loading, text processing, embeddings, vector stores, language models, prompts, and evaluation metrics:

```
from beyondllm.utils import CONTEXT_RELEVANCE, GROUNDEDNESS, ANSWER_RELEVANCE

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings    import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import re
import numpy as np
import pysbd
```

**API Keys**

Here, you'll need to replace `<your groq api key>` with your actual Groq API key to establish a connection with the language model:

```
GROQ_API_KEY = "<your groq api key>"
```

**Loading PDF Documents**

This code snippet employs the `PyPDFDirectoryLoader` class from LangChain to load PDF documents situated within a specified directory:

```
loader = PyPDFDirectoryLoader("/content/sample_data/Data")
docs = loader.load()
```

**Text Splitting**

For efficient processing, we leverage the `RecursiveCharacterTextSplitter` class to partition the loaded documents into manageable chunks. The `chunk_size` parameter controls the maximum size of each chunk, and `chunk_overlap` determines the character overlap between consecutive chunks:

```
text_splitter = RecursiveCharacterTextSplitter(chunk_size=756, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
```

**Document Embeddings**

We generate document embeddings using the `HuggingFaceEmbeddings` class. This creates numerical representations that capture the semantic meaning of each document chunk. Here, we're using the pre-trained BAAI/bge-base-en-v1.5 model:

```
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
print(embeddings)
```

**Vector Store Creation**

The document chunk embeddings are used to construct a vector store employing FAISS (Fast Approximate Nearest Neighbor Search). This facilitates efficient retrieval of documents similar to a given query based on their semantic closeness:

```
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore
```

**Querying and Retrieval**

1. **Formulate the Query:** Define a query that represents the user's information need. For instance, `query = "what causes heart diseases"`
2. **Similarity Search:** Utilize the vector store's `similarity_search` method to find documents that exhibit semantic similarity to the query. The `search_kwargs` argument allows you to configure the search parameters, such as the number of nearest neighbors (`k`) to retrieve:

```
query = "what causes heart diseases"
search = vectorstore.similarity_search(query)

# Set up the retriever for similarity search
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

retriever.invoke(query)
```

**Language Model Initialization**

We initialize a language model instance using the `ChatGroq` class from LangChain. This provides access to a powerful language model capable of generating text, translating languages, and answering questions. Remember to replace `<your groq api key>` with your actual API key:

```
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama3-8b-8192",  
    groq_api_key=GROQ_API_KEY,
    temperature=0.1  # Set the temperature to 0.1
)
```

#### Defining the Prompt Template

We'll create a prompt template using `ChatPromptTemplate` to structure the interaction between the user query and the language model. This template provides clear instructions to the model:

```
template = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI assistant that follows instruction extremely well. Please be truthful and give direct answers.<|eot_id|><|start_header_id|>user<|end_header_id|>
{query}<|eot_id|>
"""

prompt = ChatPromptTemplate.from_template(template)
```

#### Creating the RAG Chain

Now, we construct the Retrieval-Augmented Generation (RAG) chain. This chain orchestrates the retrieval of relevant documents based on the user query, formats the query and retrieved documents into a prompt, feeds it to the language model, and processes the model's response:

```
rag_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

#### Extracting Numbers from Response

A helper function to extract numerical values from the generated response:

```
def extract_number(response):
    match = re.search(r'\b(10|[0-9]+)\b', response)
    if match:
        return int(match.group(0))
    return np.nan
```

#### Tokenizing Sentences

Another helper function to split the response into sentences for further analysis:

```
def sent_tokenize(text: str):
    seg = pysbd.Segmenter(language="en", clean=False)
    return seg.segment(text)
```

### Evaluating the RAG Chain with BeyondLLM Metrics

#### Context Relevancy

This function assesses how relevant the retrieved context is to the given query:

```
def get_context_relevancy(llm, query, context):
    total_score = 0
    score_count = 0

    for content in context:
        score_response = llm.invoke(CONTEXT_RELEVANCE.format(question=query, context=content))
        
        # Access the content attribute directly
        score_str = score_response.content
        
        # Accumulate the score
        score = float(extract_number(score_str))
        total_score += score
        score_count += 1

    average_score = total_score / score_count if score_count > 0 else 0
    return f"Context Relevancy Score: {round(average_score, 1)}"
```

#### Answer Relevancy

This function evaluates how relevant the generated answer is to the given query:

```
def get_answer_relevancy(llm, query, response):
    answer_relevancy_score = llm.invoke(ANSWER_RELEVANCE.format(question=query, context=response))
    return f"Answer Relevancy Score: {answer_relevancy_score}"
```

#### Groundedness

This function assesses how grounded the generated answer is in the provided context:

```
def get_groundedness(llm, response, context):
    total_score = 0
    score_count = 0

    # Tokenize the response into sentences
    statements = sent_tokenize(response)

    for statement in statements:
        score_response = llm.invoke(GROUNDEDNESS.format(statement=statement, context=" ".join(context)))
        
        # Access the content attribute directly
        score_str = score_response.content
        
        # Accumulate the score
        score = float(extract_number(score_str))
        total_score += score
        score_count += 1

    average_groundedness = total_score / score_count if score_count > 0 else 0
    return f"Groundedness Score: {round(average_groundedness, 1)}"
```

#### Example Usage

```
# Example query
query = "what causes heart diseases?"

# Retrieve relevant documents based on the user query
retrieved_docs = retriever.invoke(query)

# Prepare the context from the retrieved documents
context = [doc.page_content for doc in retrieved_docs]

# Get context relevancy score
print(get_context_relevancy(llm, query, context))

# Generate response using RAG chain
response = rag_chain.invoke(query)

# Get answer relevancy score
answer_relevancy_score = llm.invoke(ANSWER_RELEVANCE.format(question=query, context=response))
print(answer_relevancy_score.content)

# Get groundedness score
print(get_groundedness(llm, response, context))
```

This will give us the following output:

```
Context Relevancy Score: 7.7
Answer Relevancy Score: 9 
Groundedness Score: 7.9
```

This way, we combine BeyondLLM's evaluation capabilities with LangChain's RAG framework, effectively assess the quality of generated responses based on context relevancy, answer relevancy, and groundedness.
