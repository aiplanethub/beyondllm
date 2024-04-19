# Chat with your Powerpoint file + Gradio Inference

## Import the required libraries

```python
from beyondllm import source,retrieve,embeddings,llms,generator
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

We have the default Embedding Model which is Gemini Embeddings in this case.

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

### Deploy Inference - Gradio
```python
import gradio as gr

def predict(message, history, system_prompt, tokens):
  response =  pipeline.call()
  return response

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def predict(message, chat_history):
      response = pipeline.call()
      chat_history.append((message, response))
      return "", chat_history


    msg.submit(predict, [msg, chatbot], [msg, chatbot])

demo.launch(share = True)
```
