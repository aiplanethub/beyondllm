# ⏰ Observability

> Note: Beyondllm are currently only supports observability for OpenAI models as of now

Observability is required to monitor and evaluate the performance and behaviour of your pipeline. Some key features that observability offer are:&#x20;

* **Tracking metrics:** This includes things like response time, token usage and the kind of api call (embedding, llm, etc).
* **Analyzing input and output:** Looking at the prompts users provide and the responses the LLM generates can provide valuable insights.

Overall, LLM observability is a crucial practice for anyone developing or using large language models. It helps to ensure that these powerful tools are reliable, effective, and monitored.&#x20;

Beyondllm offer observability layer with the help of [Phoenix](https://phoenix.arize.com/). We have integrated [phoenix](https://phoenix.arize.com/) within our library so you can run the dashboard with just a single command.&#x20;

```python
from beyondllm import observe
```

First you import the observe module from beyondllm&#x20;

```python
Observe = observe.Observer()
```

You then make an object of the observe.Observer()

```
Observe.run()
```

You then run the Observe object and Voila you have your dashboard running. Whatever api call you make will be reflected on your dashboard.&#x20;

<figure><img src="../.gitbook/assets/Screenshot 2024-06-05 at 00.27.55.png" alt=""><figcaption><p>Dashboard </p></figcaption></figure>

<figure><img src="../.gitbook/assets/Screenshot 2024-06-05 at 00.27.41.png" alt=""><figcaption></figcaption></figure>

### Example Snippet

```python
from beyondllm import source,retrieve,generator, llms, embeddings
from beyondllm.observe import Observer
import os

os.environ['OPENAI_API_KEY'] = 'sk-****'

Observe = Observer()
Observe.run()

llm=llms.ChatOpenAIModel()
embed_model = embeddings.OpenAIEmbeddings()

data = source.fit("https://medium.aiplanet.com/introducing-beyondllm-094902a252e2",dtype="url",chunk_size=512,chunk_overlap=50)
retriever = retrieve.auto_retriever(data,embed_model,type="normal",top_k=4)

pipeline = generator.Generate(question="why use BeyondLLM?",retriever=retriever, llm=llm)
```
