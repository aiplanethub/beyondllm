import os
import yaml

class BaseEmbeddings:
    def __init__(self,config_file=None):
        if config_file:
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
                
        else:
            pass

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
