from .base import BaseLoader
from llama_index.core.node_parser import SentenceSplitter
import subprocess
import sys
try:
    from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
except ImportError:
    user_agree = input("The feature you're trying to use requires an additional library(s):youtube_transcript_api,llama-index-readers-youtube-transcript. Would you like to install it now? [y/N]: ")
    if user_agree.lower() == 'y':
        subprocess.check_call([sys.executable, "-m", "pip", "install", "youtube_transcript_api"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-index-readers-youtube-transcript"])
        from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
    else:
        raise ImportError("The required 'llama-index-readers-youtube-transcript' is not installed.")
from dataclasses import dataclass

@dataclass
class YoutubeLoader(BaseLoader):
    chunk_size: int = 512
    chunk_overlap: int = 100

    def load(self, path):
        """Load youtube video transcript data from the URL of the video."""
        input_files = path if isinstance(path, list) else [path]
        loader = YoutubeTranscriptReader()
        docs = loader.load_data(
            ytlinks=input_files
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