from beyondllm.loaders.base import BaseLoader
from llama_index.core.node_parser import SentenceSplitter
import subprocess
import sys
import os

try:
    from llama_index.readers.notion import NotionPageReader
except ImportError:
    user_agree = input("The feature you're trying to use requires an additional library(s):llama-index-readers-notion. Would you like to install it now? [y/N]: ")
    if user_agree.lower() == 'y':
        subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-index-readers-notion"])
        from llama_index.readers.notion import NotionPageReader
    else:
        raise ImportError("The required 'llama-index-readers-notion' is not installed.")
    
from dataclasses import dataclass

@dataclass
class NotionLoader(BaseLoader):
    notion_integration_token: str = "secret_"
    chunk_size: int = 512
    chunk_overlap: int = 100

    def load(self, path):
        """Load Notion page data from the page ID of your Notion page or a list of page IDs."""
        integration_token = self.notion_integration_token or os.getenv('NOTION_INTEGRATION_TOKEN')
        page_ids = []

        if isinstance(path, str):
            page_ids.append(path)
        elif isinstance(path, list):
            page_ids.extend(path)

        loader = NotionPageReader(integration_token=integration_token)
        docs = loader.load_data(page_ids=page_ids)
        return docs

    def split(self, documents):
        """Chunk the loaded document based on size and overlap"""
        
        splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        split_documents = splitter.get_nodes_from_documents(documents)
        return split_documents

    def fit(self, path):
        """Load and split the document, then return the split parts. Uses base implementation."""
        return super().fit(path)