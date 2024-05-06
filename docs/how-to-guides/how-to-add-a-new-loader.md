# ➕ How to add a new Loader?

In building a RAG pipeline, the initial phase involves sourcing data from various origins and preparing it for usability. This process comprises two key steps: firstly, loading the data, and subsequently, splitting or chunking it for effective handling. To incorporate a new loader, adhere to these three common practices:

1. Identify and define the specific type of loader using the llama index module.
2. Configure the parameters of the loader accordingly.
3. Utilize the fit function for subsequent data processing tasks.

Here's an example of how to add a new LLM, for your Notion Pages.

{% hint style="info" %}
Note: Each Loader has its own documentation. We should refer to their documentation to learn how to use them.
{% endhint %}

## Configure Parameters

Incorporating a new loader into the RAG pipeline requires consideration of the necessary configurations and user inputs. To achieve this, we define a `dataclass` that encapsulates the parameters required for configuring the loader. Within the load function, we typically initialize the loader, ensuring its readiness for subsequent operations. Additionally, if the loader necessitates retrieving a **secret token** from an environment variable, such configuration can be seamlessly handled within the `dataclass`. This standardized format ensures consistency across various loaders, such as `urlLoader` and `youtubeLoader`.

```
from .base import BaseLoader
from llama_index.core.node_parser import SentenceSplitter
import subprocess
import sys
import os   
from dataclasses import dataclass

@dataclass
class NotionLoader(BaseLoader):
    notion_integration_token: str = "secret_" # put your notion secret token here
    chunk_size: int = 512
    chunk_overlap: int = 100
```

## Initialize the loader

The `load` function in the Enterprise RAG utilizes the llama index loaders. Here, in this case, it is the `NotionPageReader` that is being used.

```
def load(self, path):
    """Load Notion page data from the page ID of your Notion page: The hash value at the end of your URL"""
    integration_token = self.notion_integration_token or os.getenv('NOTION_INTEGRATION_TOKEN')
    loader = NotionPageReader(integration_token=integration_token)
    docs = loader.load_data(
        page_ids=[path]
    )
    return docs
```

## Split the Document

The `split` method divides the loaded document into smaller chunks based on specified size and overlap parameters, allowing efficient processing.

```
def split(self, documents):
    """Chunk the loaded document based on size and overlap"""
    
    splitter = SentenceSplitter(
        chunk_size=self.chunk_size,
        chunk_overlap=self.chunk_overlap,
    )
    split_documents = splitter.get_nodes_from_documents(documents)
    return split_documents
```

## Implement the Loader

This method combines all the different methods within the dataclass and uses the base implementation to execute the loader.

```
def fit(self, path):
    """Load and split the document, then return the split parts. Uses base implementation."""
    return super().fit(path)
```

