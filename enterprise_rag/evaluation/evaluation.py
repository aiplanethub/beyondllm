try:
    import re
    from typing import List
    import pysbd
    import numpy as np
except ImportError as e:
    missing_package = str(e).split("'")[-2]  
    raise ImportError(f"{missing_package} is required for this module. Please install it with `pip install {missing_package}`.")

from enterprise_rag.evaluation.prompts import CONTEXT_RELEVENCE, ANSWER_RELEVENCE, GROUNDEDNESS

from enterprise_rag.llms import ChatOpenAIModel


class Evaluation:
    def __init__(self, evaluater, auto_retriever, llm):
        self.evaluater = evaluater
        self.auto_retriever = auto_retriever
        self.llm = llm

    
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