import os
import yaml
from dataclasses import dataclass

@dataclass
class BaseLoader:
    """
    Base class for data loaders. This class provides the template methods load, split, and fit
    that must be implemented by subclasses to handle specific data loading and processing tasks.
    """
    path: str
    chunk_size: int = 512
    chunk_overlap: int = 100

    def load(self):
        """Load data from a given path. To be implemented by subclasses."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def split(self):
        """Split the loaded document into nodes. To be implemented by subclasses."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def fit(self, path):
        """
        Loads and splits the document, then returns the split parts.
        This method leverages the load and split methods.
        
        Parameters:
            path (str): The path to the data file.
            
        Returns:
            List[TextNode]: The split nodes of the loaded document.

        Example:
        from enterprise_rag.source import fit
        pdf_data = fit("/your/pdf/path.pdf",loader_type='pdf',chunk_size=512,chunk_overlap=50)
        youtube_data = fit(path="your-youtube-url",loader_type='youtube')
        """
        documents = self.load(path)
        split_documents = self.split(documents)
        return split_documents

