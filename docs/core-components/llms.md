# 🧠 LLMs

## What is LLMs aka Large Language Models?

An LLM, or Large Language Model, is a fundamental element of BeyondLLM. It is used in the generate function to generate a response. We support a variety of models including `ChatOpenAI`, `Gemini`, `HuggingFaceHub Models`, `AzureChatOpenAI` and `Ollama` wrapper.

### GeminiModel

Gemini is the default model used in BeyondLLM. This model includes the Gemini family models from Google.

_Notes: Currently we only support **gemini-pro** and **gemini-1.0-pro. Also no need to install Google Generative AI, because this is a default model.**_&#x20;

**Parameters**

* **Google API Key** : Key used to authenticate and access the Gemini API. Get API key from here: [https://ai.google.dev/](https://ai.google.dev/)
* **Model Name :** Defines the Gemini chat model to be used in eg: _**gemini-pro**_

**Code snippet**

```python
from beyondllm.llms import GeminiModel

llm = GeminiModel(model_name="gemini-pro",google_api_key = "<your_api_key>")
```

Import the GeminiModel from the llms and configure it according to your needs and start using it.

### ChatOpenAIModel

ChatOpenAI is a chat model provided by OpenAI which is trained on instructions dataset in a large corpus.&#x20;

**Installation**

In order to use ChatOpenAIModel, we first need to install it:

```
pip install openai
```

**Parameters**

* **OpenAI API Key**: Key used to authenticate and access the `OpenAI API`. Get your API key from here: [https://platform.openai.com/](https://platform.openai.com/docs/overview)
* **Model Name :** Defines the OpenAI chat model to be used in eg: `GPT3.5` and `GPT4` series.
* **Max Tokens :** The output sequence length response from the model.
* **Temperature** : It can be used to control the randomness or creativity in responses.

**Code snippet**

```python
from beyondllm.llms import ChatOpenAIModel

llm = ChatOpenAIModel(model="gpt-3.5-turbo",api_key = "",model_kwargs = {"max_tokens":512,"temperature":0.1})
```

Import the ChatOpenAIModel from the llms and configure it according to your needs and start using it

### HuggingFaceHubModel&#x20;

The Hugging Face Hub is a platform with over 350k models, 75k datasets, and 150k demo apps (Spaces), all open source and publicly available, in an online platform where people can easily collaborate and build ML together.

**Installation**

In order to use HuggingFaceModel, we first need to install it:

```
pip install huggingface_hub
```

**Parameters**

* **Token :** HuggingFace Access Token to run the model on Inference API. You can get your Access token from here: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
* **Model :** Model name from the HuggingFace Hub – defaults to `zephyr-7b-beta`.

**Code snippet**

<pre class="language-python"><code class="lang-python"><strong>from beyondllm.llms import HuggingFaceHubModel
</strong>
llm = HuggingFaceHubModel(model="huggingfaceh4/zephyr-7b-alpha",token="&#x3C;replace_with_your_token>",model_kwargs={"max_new_tokens":512,"temperature":0.1})
</code></pre>

Specify the model name from the HuggingFaceHub and add your token and start using it.

### Ollama&#x20;

Ollama lets you run models locally and use them in your application.

In order to get started with Ollama, we first need to download it, and pull the model based on our need.  Download Ollama: [https://ollama.com/download](https://ollama.com/download)

**Basic Ollama Commands**

```bash
ollama pull llama2 # loads llama2 model locally

ollama pull gemma # loads gemma mdoel locally

ollama list # displays all the models that are installed
```

More commands: [https://github.com/ollama/ollama](https://github.com/ollama/ollama)

**Installation**

```
pip install ollama
```

**Parameters**

* **Model** : The name of the model you are using.

**Code snippet**

Make sure, before you run the OllamaModel, the model is running locally on your terminal. `ollama run llama2`

```python
from beyondllm.llms import OllamaModel
    
llm = OllamaModel(model="llama2")
```

### AzureOpenAIModel

Azure OpenAI Service provides REST API access to OpenAI’s powerful language models including the `GPT-4`, `GPT-3.5-Turbo`, and `Embeddings model` series.

**Installation**

In order to use AzureOpenAIModel, we first need to install it:

```
pip install openai
```

**Parameters**

* **AzureChatOpenAI API Key:** Azure api key for AzureChatOpenAI service.&#x20;
* **Deployment Name :** Enter the the deployment name that is created on Model deployments on Azure
* **Endpoint Url :** Enter your endpoint url.&#x20;
* **Model Name :** AzureChatOpenAI enables the access to GPT4 models.
* **Max Tokens :** The maximum sequence length for the model response.
* **Temperature :** It can be used to control the randomness or creativity in responses.

> Create your Azure account and get Endpoint URL and Key from here: [https://oai.azure.com/](https://oai.azure.com/)

**Code snippet**

```python
from beyondllm.llms import AzureOpenAIModel

llm = AzureOpenAIModel(model="gpt4",api_key = "<your_api_key>",deployment_name="",endpoint_url="",model_kwargs={"max_tokens":512,"temperature":0.1})
```
