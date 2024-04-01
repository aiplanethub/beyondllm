from .base import BaseLoader
from llama_index.core.node_parser import SentenceSplitter
import subprocess
import sys
try:
    from llama_index.readers.web import SimpleWebPageReader
except ImportError:
    user_agree = input("The feature you're trying to use requires an additional library. Would you like to install it now? [y/N]: ")
    if user_agree.lower() == 'y':
        subprocess.check_call([sys.executable, "-m", "pip", "install", "llama_index.readers.web"])
        from llama_index.readers.web import SimpleWebPageReader
    else:
        raise ImportError("The required 'llama_index.readers.web' is not installed.")
    

class UrlLoader(BaseLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 

    def load(self, path):
        """Load web page data from a file."""
        docs = SimpleWebPageReader(html_to_text=True).load_data(
            [path]
        )
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