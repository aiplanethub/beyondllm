from beyondllm.loaders.base import BaseLoader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.node_parser import SentenceSplitter
import subprocess
import sys
try:
    from llama_index.readers.web import SimpleWebPageReader
except ImportError:
    user_agree = input("The feature you're trying to use requires an additional library:llama_index.readers.web Would you like to install it now? [y/N]: ")
    if user_agree.lower() == 'y':
        subprocess.check_call([sys.executable, "-m", "pip", "install", "llama_index.readers.web"])
        from llama_index.readers.web import SimpleWebPageReader
    else:
        raise ImportError("The required 'llama_index.readers.web' is not installed.")
from dataclasses import dataclass

@dataclass    
class UrlLoader(BaseLoader):
    chunk_size: int = 512
    chunk_overlap: int = 150

    def load(self, path):
        """Load data from a single URL or a list of URLs."""
        urls = []

        if isinstance(path, str):
            urls.append(path)
        elif isinstance(path, list):
            urls.extend(path)

        docs = SimpleWebPageReader(urls=urls).load_data()
        return docs

    def split(self, documents):
        """Chunk the loaded document based on size and overlap"""
        
        parser = MarkdownNodeParser()
        nodes = parser.get_nodes_from_documents(documents)
        splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        split_documents = splitter.get_nodes_from_documents(nodes)
        return split_documents

    def fit(self, path):
        """Load and split the document, then return the split parts. Uses base implementation."""
        return super().fit(path)