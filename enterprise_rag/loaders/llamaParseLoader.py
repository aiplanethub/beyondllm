from .base import BaseLoader
from llama_parse import LlamaParse
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import MarkdownNodeParser
import os

class LlamaParseLoader(BaseLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.llama_parse_key = kwargs.get('llama_parse_key',os.getenv('LLAMA_CLOUD_API_KEY'))


    def load(self, path):
        """Load data from a file."""
        docs = LlamaParse(result_type="markdown").load_data(path)
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