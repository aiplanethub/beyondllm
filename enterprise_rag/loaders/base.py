import os
import yaml

class BaseLoader:
    def __init__(self, chunk_size=1024, chunk_overlap=200, llama_parse_key=None, config_file=None):
        if config_file:
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
                self.chunk_size = config.get('chunk_size', chunk_size)
                self.chunk_overlap = config.get('chunk_overlap', chunk_overlap)
                self.llama_parse_key = config.get('api_key', llama_parse_key)
        else:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.llama_parse_key = llama_parse_key or os.getenv('LLAMA_PARSE_API_KEY')

    def load(self, path):
        """Load data from a given path. To be implemented by subclasses."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def split(self, document):
        """Split the loaded document into parts. To be implemented by subclasses."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def fit(self, path):
        """Load and split the document, then return the split parts."""
        documents = self.load(path)
        split_documents = self.split(documents)
        return split_documents

