def CONTEXT_RELEVENCE(context,question):
    return f""" As a RELEVANCE grader, your task is to rate the CONTEXT's RELEVANCE to the QUESTION on a scale from 0 to 10, where 0 signifies least RELEVANT and 10 the most RELEVANT. Length of the CONTEXT is not a factor in determining its RELEVANCE. Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant.
    Follow these guidelines for scoring:

    - Length Irrelevant: The length of CONTEXT does not impact its score.
    - Relevance to QUESTION: The score should reflect the extent to which the CONTEXT is RELEVANT to the QUESTION. Higher scores are awarded as the CONTEXT becomes more RELEVANT to the QUESTION.
    - CONTEXT is not similar at all to the question is given a score of 0 or at max 1.
    - Partial Relevance: Scores of 2 to 4 are assigned when the CONTEXT is RELEVANT to some parts of the QUESTION.
    - Substantial Relevance: Scores of 5 to 8 are given when the CONTEXT is significantly RELEVANT to most parts of the QUESTION.
    - Full Relevance: A score of 9 or 10 indicates that the CONTEXT is fully RELEVANT to the entire QUESTION.
    - Perfect Score: To achieve a score of 10, the CONTEXT must be completely RELEVANT and helpful in answering the entire QUESTION.

    QUESTION: {question}

    CONTEXT: {context}

    RELEVANCE:
    """

def ANSWER_RELEVENCE(response,question):
    return f"""
    You are a RELEVANCE grader; providing the relevance of the given CONTEXT that is basically a response from am AI to the given QUESTION which was asked by the user.
    Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant.

    A few additional scoring guidelines:

    - Long CONTEXTS should score equally well as short CONTEXTS.

    - RELEVANCE score should increase as the CONTEXTS provides more RELEVANT context to the QUESTION.

    - RELEVANCE score should increase as the CONTEXTS provides RELEVANT context to more parts of the QUESTION.

    - CONTEXT that is RELEVANT to some of the QUESTION should score of 2, 3 or 4. Higher score indicates more RELEVANCE.

    - CONTEXT that is RELEVANT to most of the QUESTION should get a score of 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

    - CONTEXT that is RELEVANT to the entire QUESTION should get a score of 9 or 10. Higher score indicates more RELEVANCE.

    - CONTEXT must be relevant and helpful for answering the entire QUESTION to get a score of 10.

    - Never elaborate.

    QUESTION: {question}

    CONTEXT: {response}

    RELEVANCE:
    """

def GROUNDEDNESS(statement,context):
    return f"""
    You are a GROUNDEDNESS grader, you will be given a STATEMENT and CONTEXT. Your job is to compare the two and give a score based on it. Use the SCORING SCALE below to give a SCORE
    Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant.
    Only reply in the form of a number nothing else.
    Scoring Scale:
    - 10: The CONTEXT directly supports the STATEMENT with identical or nearly identical information.
    - 7 to 9: The CONTEXT supports the STATEMENT thematically, with similar or related information, but not verbatim.
    - 4 to 6: The CONTEXT contains vague or partially relevant information, indicating a loose connection to the STATEMENT.
    - 1 to 3: The CONTEXT only has minimal and very indirect references to the STATEMENT.
    - 0: The CONTEXT does not support the STATEMENT in any discernible way; no direct, thematic, or vague connections are present.


    STATEMENT: {statement}
    CONTEXT: {context}
    Score:
    """


from enterprise_rag.utils import CONTEXT_RELEVENCE,GROUNDEDNESS,ANSWER_RELEVENCE

import re
from typing import List
import pysbd
import numpy as np

class Evaluator:
    def context_relevance(self, question):
        try:
            response = self.auto_retriever.retrieve(question)
            
            entire_context = [node_with_score.node.text for node_with_score in response]
            total_score = 0
            score_count = 0

            for context in entire_context:

                score_str = self.evaluater.predict(CONTEXT_RELEVENCE.format(question=question, context=context))
            
                try:
                    score = float(score_str)
                    total_score += score
                    score_count += 1
                except ValueError:
                    print("Invalid score format:", score_str)

            if score_count > 0:
                average_score = total_score / score_count
            else:
                average_score = 0
            return average_score
        except Exception as e:
            print(f"Failed during context relevance evaluation: {e}")
            
    
    def answer_relevance(self, question):
        try:
            response = self.llm.query(question)
            print("Response from the generator:", response.response)
            score = self.evaluater.predict(ANSWER_RELEVENCE.format(question=question, context=response.response))
            print ("Answer Relevancy Score:", score)
        
        except Exception as e:
            print(f"Failed during answer relevance evaluation: {e}")
            
    
    def groundedness(self, question):
        try:

            response = self.llm.query(question)
            print("Response from the generator:", response.response)
            statements = sent_tokenize(response.response)
            scores = []
            for statement in statements:

                score_response = self.evaluater.predict(GROUNDEDNESS.format(statement=statement, context=response.source_nodes))
                
                

                score = extract_number(score_response)

                print("Statement:", statement)
                print("Score:", score)
                scores.append(score)
                
            average_score = sum(scores) / len(scores) if scores else 0
            print("Average score:", average_score)
        
        except Exception as e:
            raise Exception("Failed during groundedness evaluation: ", str(e))


def sent_tokenize(text: str) -> List[str]:
    seg = pysbd.Segmenter(language="en", clean=False)
    return seg.segment(text)

def extract_number(response):
    match = re.search(r'\b(10|[0-9])\b', response)
    if match:
        return int(match.group(0))
    return np.nan