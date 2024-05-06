# ➕ How to add new Embeddings?

Embeddings are involved when similar documents to the user query need to be produced. Various embeddings models are available, both closed source and open source. For Enterprise, we use embeddings supported by LlamaIndex. To add new embeddings, four important things need to be added:

1. Configurable parameters
2. Loading the base embedding model with the parameters it supports, such as model name, API key, and so on.
3. Embedding text to embeddings, to convert text to embeddings
4. Batching and aggregation supporting functions that are used to evaluate the embeddings.

### Config parameters

Define the dataclass that takes the parameters required to load the model. If an API key needs to be set in an environment variable, it should be added in the `__post_init__` function. Some parameters may have default values, while others may be left as undefined.&#x20;

E.g., model\_name can either be initialised as:

* `model_name: str`
* `model_name: str = field(default='BAAI/bge-small-en-v1.5')`

{% hint style="info" %}
Note: `load_from_kwargs` is a default static method, that is used in every Embedding model. This ensures that the user can enter any parameters that is supported by that embedding model.&#x20;
{% endhint %}

```python
from .base import BaseEmbeddings,EmbeddingConfig
from typing import Any, Optional
from dataclasses import dataclass,field
import warnings
warnings.filterwarnings("ignore")

@dataclass
class FastEmbedEmbeddings:
    """
    from enterprise_rag.embeddings import FastEmbedEmbeddings
    embed_model = FastEmbedEmbeddings()
    """
    model_name:  str = field(default='BAAI/bge-small-en-v1.5')

    def __post_init__(self):
        self.load()
        
    @staticmethod
    def load_from_kwargs(self,kwargs): 
        embed_config = EmbeddingConfig(**kwargs)
        self.config = embed_config
        self.load()
```

### Load Embedding Model

The `load` function in Enterprise RAG utilizes LlamaIndex embeddings. It simply initializes the embedding model.

```python
def load(self):
    try:
        from llama_index.embeddings.fastembed import FastEmbedEmbedding
    except:
        raise ImportError("Qdrant Fast Embeddings Embeddings library is not installed. Please install it with ``pip install llama-index-embeddings-fastembed``.")
    
    try:
        self.client = FastEmbedEmbedding(model_name=self.model_name)

    except Exception as e:
        raise Exception("Failed to load the embeddings from Fast Embeddings:", str(e))
    
    return self.client
```

### Embed Text

The `embed_text` function is simply used to convert a normal string into embeddings.

```python
def embed_text(self,text):
    embeds = self.client.get_text_embedding(text) 
    return embeds
```

### Supporting batching and agg embedding functions

These supporting functions are necessary for batching queries during evaluation. As we evaluate the Embedding models to obtain `hit rate` and `mean reciprocal rank (MRR)`, we need to generate question and answer (Q\&A) pairs. The `get_agg_embedding_from_queries` function facilitates this process. To pass batches of queries to the embeddings, we utilize the `get_text_embedding_batch` function.

```python
def get_text_embedding_batch(self,*args, **kwargs):
    batch = self.client.get_text_embedding_batch(*args, **kwargs)
    return batch

def get_agg_embedding_from_queries(self,*args,**kwargs):
    agg_embedding = self.client.get_agg_embedding_from_queries(*args, **kwargs)
    return agg_embedding
```
