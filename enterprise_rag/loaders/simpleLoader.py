from .base import BaseLoader
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from dataclasses import dataclass

@dataclass
class SimpleLoader(BaseLoader):
    """
    A loader class for handling simple file types. Inherits from BaseLoader and implements
    the load and split methods for basic file processing.
    """
    chunk_size: int = 512
    chunk_overlap: int = 100

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs) 

    def load(self, path):
        """Load data from a file."""
        docs = SimpleDirectoryReader(input_files=[path]).load_data()
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