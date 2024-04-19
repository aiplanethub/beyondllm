from beyondllm.loaders.base import BaseLoader
from llama_parse import LlamaParse
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import MarkdownNodeParser
import os
from dataclasses import dataclass

@dataclass
class LlamaParseLoader(BaseLoader):
    llama_parse_key: str = "llx-"
    chunk_size: int = 512
    chunk_overlap: int = 100
    
    def load(self, path):
        """Load data from a file to be parsed by LlamaParse cloud API."""
        llama_parse_key = self.llama_parse_key or os.getenv('LLAMA_CLOUD_API_KEY')
        try:
            docs = LlamaParse(result_type="markdown",api_key=llama_parse_key).load_data(path)
        except:
            raise ValueError("File not compatible/no result returned from Llamaparse")
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