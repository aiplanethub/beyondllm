## IGNORE, this needs embedding model, can't give embed in fit function

from .base import BaseLoader
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
import os
from dataclasses import dataclass

@dataclass
class SemanticChunkingLoader(BaseLoader):
    
    def load(self, path):
        """Load data from a file."""
        try:
            docs =  SimpleDirectoryReader(path).load_data()
        except:
            raise 
        return docs

    def split(self, documents):
        """Chunk the loaded document based on size and overlap"""
        
        splitter = SemanticSplitterNodeParser(
            buffer_size=self.buffer_size, breakpoint_percentile_threshold=self.breakpoint,
        )
        split_documents = splitter.get_nodes_from_documents(documents)
        return split_documents

    def fit(self, path):
        """Load and split the document, then return the split parts. Uses base implementation."""
        return super().fit(path)