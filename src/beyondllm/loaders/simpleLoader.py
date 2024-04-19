from beyondllm.loaders.base import BaseLoader
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from dataclasses import dataclass

@dataclass
class SimpleLoader(BaseLoader):
    """
    A loader class for handling simple file types. Inherits from BaseLoader and implements
    the load and split methods for basic file processing. 

    Supported file types: pdf", "csv", "docx", "epub", "md", "ppt", "pptx", "pptm"
    """
    chunk_size: int = 512
    chunk_overlap: int = 150

    def load(self, path):
        """Load data from a file."""
        docs = SimpleDirectoryReader(input_files=[path]).load_data()
        return docs

    def split(self, documents):
        """
        Chunk the loaded document based on size and overlap. 
        Recursively splits data and tries to keep sentences and paragraphs whole
        """
        splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        split_documents = splitter.get_nodes_from_documents(documents)
        return split_documents

    def fit(self, path):
        """Load and split the document, then return the split parts. Uses base implementation."""
        return super().fit(path)