from beyondllm.llms import BaseLLMModel

import numpy as np
import re
import uuid
from typing import List

from tqdm import tqdm

from llama_index.core.schema import MetadataMode, TextNode
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

DEFAULT_QA_GENERATE_PROMPT_TMPL = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided."
"""


def generate_qa_embedding_pairs(
    nodes: List[TextNode],
    llm: BaseLLMModel,
    qa_generate_prompt_tmpl: str = DEFAULT_QA_GENERATE_PROMPT_TMPL,
    num_questions_per_chunk: int = 2,
) -> EmbeddingQAFinetuneDataset:
    """Generate examples given a set of nodes."""
    node_dict = {
        node.node_id: node.get_content(metadata_mode=MetadataMode.NONE)
        for node in nodes
    }

    queries = {}
    relevant_docs = {}
    for node_id, text in tqdm(node_dict.items()):
        query = qa_generate_prompt_tmpl.format(
            context_str=text, num_questions_per_chunk=num_questions_per_chunk
        )
        response = llm.predict(query)

        result = str(response).strip().split("\n")
        questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ]
        questions = [question for question in questions if len(question) > 0]

        for question in questions:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            relevant_docs[question_id] = [node_id]

    return EmbeddingQAFinetuneDataset(
        queries=queries, corpus=node_dict, relevant_docs=relevant_docs
    )

def generate_qa_dataset(llm,nodes):
    print("Generating QA dataset....")

    qa_dataset = generate_qa_embedding_pairs(
        llm=llm,
        nodes=nodes,
        qa_generate_prompt_tmpl = DEFAULT_QA_GENERATE_PROMPT_TMPL,
        num_questions_per_chunk=2
    )

    return qa_dataset


def evaluate_from_dataset(dataset, retriever):
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    hits = 0
    eval_results = []

    for query_id, query in queries.items():
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

def evaluate_retriever(llm, nodes, retriever):
    qa_dataset = generate_qa_dataset(llm,nodes)
    hit_rate, mrr = evaluate_from_dataset(qa_dataset,retriever)
    return f"Hit_rate:{hit_rate}\nMRR:{mrr}"