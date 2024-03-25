from llama_index.legacy.finetuning import generate_qa_embedding_pairs

import pandas as pd
from llama_index.core import SimpleDirectoryReader, ServiceContext
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
)
from typing import List
from llama_index.core.indices.query.schema import QueryBundle, QueryType
from llama_index.core.schema import NodeWithScore
from llama_index.core.evaluation import RetrieverEvaluator

import time
import json
import numpy as np
import pandas as pd
import os

from tqdm.notebook import tqdm
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex

def generate_qa_dataset(llm,nodes):
    prompt = """\
    Context information is below.

    ---------------------
    {context_str}
    ---------------------

    Given the context information and not prior knowledge, generate only questions based on the below query.

    You are a Teacher/ Professor. Your task is to setup {num_questions_per_chunk} questions for an upcoming quiz/examination.
    The questions should be diverse in nature across the document. Restrict the questions to the context information provided."
    """

    qa_dataset = generate_qa_embedding_pairs(
        llm=llm,
        nodes=nodes,
        qa_generate_prompt_tmpl = prompt,
        num_questions_per_chunk=2
    )

    return qa_dataset


def evaluate(nodes, dataset, embed_model, top_k=5):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
    print(nodes)
    index = VectorStoreIndex(
        nodes, embed_model=embed_model
    )
    retriever = index.as_retriever(similarity_top_k=top_k)

    hits = 0
    eval_results = []

    for query_id, query in tqdm(queries.items()):
        retrieved_nodes = retriever.retrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]

        expected_id = relevant_docs[query_id][0]
        is_hit = expected_id in retrieved_ids

        if is_hit:
            hits += 1
            rank = retrieved_ids.index(expected_id) + 1
            mrr = 1 / rank
        else:
            mrr = 0
        eval_results.append(mrr)

    hit_rate = hits / len(queries)
    average_mrr = np.average(eval_results)

    return hit_rate, average_mrr
