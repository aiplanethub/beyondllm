# 🧠 Memory

### What is Memory?

In the Beyond LLM framework, memory is a vital component that allows language models to retain context from previous interactions. This capability enhances the model's ability to generate responses that are not only relevant to the current input but also informed by past conversations. Memory facilitates a more engaging and personalized user experience, making it particularly useful for applications such as chatbots, virtual assistants, and interactive storytelling.

### Code Snippet

### Importing Necessary Components

To use the memory functionality in Beyond LLM, you need to import the relevant classes. Below is the code snippet for importing the necessary components:

```python
from beyondllm.memory import ChatBufferMemory
```

### Basic Implementation of Memory

Once you have imported the necessary components, you can implement memory in your application. Here’s how to initialize memory and use it in a conversation:

```python
# Initialize memory with a specified window size
memory = ChatBufferMemory(window_size=3)  # Retains the last three interactions

# Initialize the language model
llm = GPT4oOpenAIModel(model="gpt-4o", api_key="sk-proj-xxxxxxxx")

# Define a function to handle the conversation
def ask_question(question):
    # Create a retriever for the sourced data (this should be defined earlier in your code)
    retriever = retrieve.auto_retriever(
        data=data,  # Ensure 'data' is defined with your source content
        type='normal',
        embed_model=embed_model,
        top_k=4,
    )
    
    # Generate a response using the memory
    pipeline = Generate(retriever=retriever, question=question, llm=llm, memory=memory, system_prompt="Answer the user questions based on the chat history")
    response = pipeline.call()
    return response

# Example interaction
response = ask_question("My name is Rupert Grint.")
print("Response:", response)

# Access the memory content after the conversation
print("\nMemory:", memory.get_memory())
```

### Example Usage

Here’s a more complete example that includes sourcing data and simulating a conversation:

```python
# Set up the embedding model
embed_model = GeminiEmbeddings(model_name="models/embedding-001", api_key="xxxxxxx-9q0nQUoM")

# Source data from a YouTube video
data = source.fit("https://www.youtube.com/watch?v=xJvclySzrSA&pp=ygUQaG9nd2FydHMgaGlzdG9yeQ%3D%3D", dtype="youtube", chunk_size=512, chunk_overlap=50)

# Initialize memory with a specified window size
memory = ChatBufferMemory(window_size=3)  # Retains the last three interactions

# Initialize the language model
llm = GPT4oOpenAIModel(model="gpt-4o", api_key="sk-proj-xxxxxxxx")

# Define a function to handle the conversation
def ask_question(question):
    # Create a retriever for the sourced data
    retriever = retrieve.auto_retriever(
        data=data,
        type='normal',
        embed_model=embed_model,
        top_k=4,
    )
    
    # Generate a response using the memory
    pipeline = Generate(retriever=retriever, question=question, llm=llm, memory=memory, system_prompt="Answer the user questions based on the chat history")
    response = pipeline.call()
    return response

# Example conversation
questions = [
    "My name is Rupert Grint.",
    "I studied in Hogwarts. Do you know that place?",
    "My best friends are Harry and Emma. Do you think I need more friends?",
    "What all did I just tell you about myself? What did I ask before introducing myself?"
]

# Loop through the questions and print responses
for question in questions:
    response = ask_question(question)
    print(f"Response: {response}")

# Access the memory content after the conversation
print("\nMemory:", memory.get_memory())
```

### Conclusion

The memory component in Beyond LLM is essential for creating interactive and personalized applications. By enabling the model to remember and utilize past interactions, it enhances the overall functionality and user engagement of LLM-powered systems. This implementation serves as a practical guide for integrating memory into your applications, ensuring that the interactions remain coherent and contextually relevant.ShareRewrite\
