from enterprise_rag.llms import ChatOpenAIModel
from enterprise_rag.utils import CONTEXT_RELEVENCE,GROUNDEDNESS,ANSWER_RELEVENCE, GROUND_TRUTH

import os,re
import numpy as np
from typing import List
import pysbd
from .base import BaseGenerator,GeneratorConfig
from dataclasses import dataclass,field

def default_llm():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY must be set in the environment variables.")
    return ChatOpenAIModel(api_key=api_key)

def extract_number(response):
    match = re.search(r'\b(10|[0-9])\b', response)
    if match:
        return int(match.group(0))
    return np.nan

def sent_tokenize(text: str) -> List[str]:
    seg = pysbd.Segmenter(language="en", clean=False)
    return seg.segment(text)

@dataclass
class Generate:
    """
    Example:
    from enterprise_rag.generator import generate
    pipeline = generate(retriever=retriever,question) #llm = ChatOpenAI
    # for prediction:
    response = pipeline.call() # rag response
    rag_triad = pipeline.rag_triad_evals_report()
    context_relevancy = pipeline.get_context_relevancy_score()
    answer_relevancy = pipeline.get_answer_relevancy_score()
    groundness = 
    """
    question: str
    retriever:str = ''
    llm: ChatOpenAIModel = field(default_factory=default_llm)
    
    def __post_init__(self):
        self.pipeline()

    def pipeline(self):
        self.CONTEXT = ""
        for node in self.retriever.retrieve(self.question):
            self.CONTEXT += node.node.text
        
        template = f"""
        You are an AI assistant who always answer to the user QUERY within the given CONTEXT \
        You are only job here it to act as knowledge transfer and given accurate response to QUERY by looking in depth into CONTEXT \
        If QUERY is not with the context, YOU MUST tell `I don't know` \
        YOU MUST not hallucinate \
        If you FAIL to execute this task, you will be fired and you will suffer \

        CONTEXT: {self.CONTEXT }
        QUERY: {self.question}
        """

        self.RESPONSE = self.llm.predict(template)
        return self.CONTEXT,self.RESPONSE

    def call(self):
        return self.RESPONSE

    def get_rag_triad_evals(self):
        context_relevancy = self.get_context_relevancy()
        answer_relevancy = self.get_answer_relevancy()
        groundness = self.get_groundness()
        return context_relevancy,answer_relevancy,groundness
        
    def get_context_relevancy(self):
        total_score = 0
        score_count = 0
        
        for context in self.CONTEXT:
            score_str = self.llm.predict(CONTEXT_RELEVENCE.format(question=self.question, context=context))
            score = float(extract_number(score_str))
            total_score += score
            score_count += 1

        if score_count > 0:
            average_score = total_score / score_count
        else:
            average_score = 0

        return f"Context relevancy Score: {average_score}"

    def get_answer_relevancy(self):
        try:
            score_str = self.llm.predict(ANSWER_RELEVENCE.format(question=self.question, context= self.RESPONSE))
            score = float(extract_number(score_str))
            return f"Answer relevancy Score: {score}"
        except Exception as e:
            print(f"Failed during answer relevance evaluation: {e}")
        
    def get_groundness(self):
        try:
            statements = sent_tokenize(self.RESPONSE)
            scores = []
            for statement in statements:

                score_response = self.llm.predict(GROUNDEDNESS.format(statement=statement, context=self.CONTEXT))
                score = extract_number(score_response)

                print("Statement:", statement)
                print("Score:", score)
                scores.append(score)
                
            average_score = sum(scores) / len(scores) if scores else 0
            return f"Groundness score: {average_score}"
        
        except Exception as e:
            raise Exception("Failed during groundedness evaluation: ", str(e))

    def get_ground_truth(self, answer:str):
        if answer is None:
            print("Error: Missing 'answer' parameter. Usage: get_ground_truth(answer=\"your_answer_here\").")
            return
        score_str = self.llm.predict(GROUND_TRUTH.format(ground_truth=answer, generated_response=self.RESPONSE))
        score = extract_number(score_str)
        return f"Ground truth score: {score}"
        

    @staticmethod
    def load_from_kwargs(self,kwargs): 
        model_config = GeneratorConfig(**kwargs)
        self.config = model_config
        self.pipeline()
