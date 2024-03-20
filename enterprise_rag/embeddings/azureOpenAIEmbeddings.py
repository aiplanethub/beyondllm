from .base import BaseEmbeddings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
# from typing import Any, Callable, Coroutine, List, Optional, Tuple
# Embedding = List[float]
# from llama_index.core.instrumentation.events.embedding import (
#     EmbeddingEndEvent,
#     EmbeddingStartEvent,
# )
# from llama_index.core.utils import get_tqdm_iterable
# from llama_index.core.callbacks.schema import CBEventType, EventPayload
# import llama_index.core.instrumentation as instrument

# dispatcher = instrument.get_dispatcher(__name__)

import os

class AzureEmbeddings(BaseEmbeddings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.azure_endpoint = kwargs.get('azure_endpoint', os.getenv('AZURE_EMBEDDINGS_ENDPOINT'))
        self.azure_key = kwargs.get('azure_key', os.getenv('AZURE_EMBEDDINGS_KEY'))
        self.api_version = kwargs.get('api_version')
        self.azure_deployment_name = kwargs.get('azure_deployment_name')

    def get_embeddings(self):
        embed_model = AzureOpenAIEmbedding(
            api_key=self.azure_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
        )
        return embed_model
    
    # def get_text_embedding_batch(
    #     self,
    #     texts: List[str],
    #     show_progress: bool = False,
    #     **kwargs: Any,
    # ) -> List[Embedding]:
    #     """Get a list of text embeddings, with batching."""
    #     cur_batch: List[str] = []
    #     result_embeddings: List[Embedding] = []

    #     queue_with_progress = enumerate(
    #         get_tqdm_iterable(texts, show_progress, "Generating embeddings")
    #     )

    #     for idx, text in queue_with_progress:
    #         cur_batch.append(text)
    #         if idx == len(texts) - 1 or len(cur_batch) == self.embed_batch_size:
    #             # flush
    #             dispatcher.event(EmbeddingStartEvent(model_dict=self.to_dict()))
    #             with self.callback_manager.event(
    #                 CBEventType.EMBEDDING,
    #                 payload={EventPayload.SERIALIZED: self.to_dict()},
    #             ) as event:
    #                 embeddings = self._get_text_embeddings(cur_batch)
    #                 result_embeddings.extend(embeddings)
    #                 event.on_end(
    #                     payload={
    #                         EventPayload.CHUNKS: cur_batch,
    #                         EventPayload.EMBEDDINGS: embeddings,
    #                     },
    #                 )
    #             dispatcher.event(
    #                 EmbeddingEndEvent(chunks=cur_batch, embeddings=embeddings)
    #             )
    #             cur_batch = []

    #     return result_embeddings