from .base import BaseLoader
from llama_index.core.node_parser import SentenceSplitter
import subprocess
import sys

try:
    from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
except ImportError:
    user_agree = input("The feature you're trying to use requires an additional library. Would you like to install it now? [y/N]: ")
    if user_agree.lower() == 'y':
        subprocess.check_call([sys.executable, "-m", "pip", "install", "youtube_transcript_api"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-index-readers-youtube-transcript"])
        from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
    else:
        raise ImportError("The required 'llama_index.readers.web' is not installed.")
    

class YoutubeLoader(BaseLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 

    def load(self, path):
        """Load web page data from a file."""
        loader = YoutubeTranscriptReader()
        docs = loader.load_data(
            ytlinks=[path]
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