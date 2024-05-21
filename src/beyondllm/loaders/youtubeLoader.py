from beyondllm.loaders.base import BaseLoader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import MarkdownNodeParser
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
    chunk_overlap: int = 150

    def load(self, path):
        """Load YouTube video transcript data from a single URL or a list of URLs."""
        ytlinks = []

        if isinstance(path, str):
            ytlinks.append(path)
        elif isinstance(path, list):
            ytlinks.extend(path)

        print(ytlinks)

        loader = YoutubeTranscriptReader()
        docs = loader.load_data(ytlinks=ytlinks)
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