from llama_index.legacy.finetuning import generate_qa_embedding_pairs
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd


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
    print("Generating QA dataset....")

    qa_dataset = generate_qa_embedding_pairs(
        llm=llm,
        nodes=nodes,
        qa_generate_prompt_tmpl = prompt,
        num_questions_per_chunk=2
    )

    return qa_dataset


def evaluate_from_dataset(dataset, retriever):
    # corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    # nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
    
    # index = VectorStoreIndex(
    #     nodes, embed_model=embed_model
    # )
    # retriever = index.as_retriever(similarity_top_k=top_k)

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
    return {"hit_rate":hit_rate,"mrr":mrr}
