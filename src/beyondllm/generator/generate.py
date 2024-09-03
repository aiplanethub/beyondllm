import os
import re
import warnings
import numpy as np
from typing import List, Dict
import pysbd
from dataclasses import dataclass, field

from beyondllm.llms import GeminiModel
from beyondllm.utils import CONTEXT_RELEVENCE, GROUNDEDNESS, ANSWER_RELEVENCE, GROUND_TRUTH
from beyondllm.memory.base import BaseMemory
from .base import BaseGenerator, GeneratorConfig


warnings.filterwarnings('always', category=DeprecationWarning, module=__name__)

def default_llm():
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY must be set in the environment variables.")
    return GeminiModel(google_api_key=api_key)

def extract_number(response):
    match = re.search(r'\b(10|[0-9])\b', response)
    if match:
        return int(match.group(0))
    return np.nan

def sent_tokenize(text: str) -> List[str]:
    seg = pysbd.Segmenter(language="en", clean=False)
    return seg.segment(text)

def thresholdCheck(score):
    if score >= 8:
        return "This response meets the evaluation threshold. It demonstrates strong comprehension and coherence."
    else:
        return "This response does not meet the evaluation threshold. Consider refining the structure and content for better clarity and effectiveness."

@dataclass
class Generate:
    """
    ### Example for new usage:
    ```
    >>> from beyondllm.generator import Generate
    >>> generator = Generate(retriever=retriever)
    >>> response = generator.call("What is the capital of France?")
    ```
    ### Example for old usage (deprecated):
    ```
    >>> from beyondllm.generator import Generate
    >>> generator = Generate(retriever=retriever, question="What is the capital of France?")
    >>> response = generator.call()
    ```
    ### for evaluation:
    ```
    >>> rag_triad = generator.rag_triad_evals()
    >>> context_relevancy = generator.get_context_relevancy()
    >>> answer_relevancy = generator.get_answer_relevancy()
    >>> groundness = generator.get_groundedness()
    ```
    """
    question: str = None
    system_prompt: str = None
    retriever: str = ''
    llm: GeminiModel = field(default_factory=default_llm)
    memory: BaseMemory = None

    def __post_init__(self):
        if self.question is not None:
            warnings.warn(
                "Initializing Generate with 'question' is deprecated and will be removed in a future version. "
                "Use the new 'call' method with a question parameter instead.",
                DeprecationWarning,
                stacklevel=2
            )
            self._legacy_init()
        
        if self.system_prompt is None:
            self.system_prompt = """
            You are an AI assistant who always answer to the user QUERY within the given CONTEXT \
            You are only job here it to act as knowledge transfer and given accurate response to QUERY by looking in depth into CONTEXT. \
            Use both the context and CHAT HISTORY to answer the query. \
            If QUERY is not within the context or chat history, YOU MUST tell `I don't know. The query is not in the given context` \
            YOU MUST not hallucinate. You are best when it comes to answering from CONTEXT \
            If you FAIL to execute this task, you will be fired and you will suffer 
            """

    def _legacy_init(self):
        self.CONTEXT = [node_with_score.node.text for node_with_score in self.retriever.retrieve(self.question)]
        self.RESPONSE = self._generate_response(self.question)

    def call(self, question: str = None) -> str:
        if question is None:
            if self.question is None:
                raise ValueError("No question provided. Either initialize with a question or pass one to call().")
            warnings.warn(
                "Using call() without a question parameter is deprecated and will be removed in a future version. "
                "Please use call(question) instead.",
                DeprecationWarning,
                stacklevel=2
            )
            return self.RESPONSE
        
        self.question = question
        self.CONTEXT = [node_with_score.node.text for node_with_score in self.retriever.retrieve(self.question)]
        self.RESPONSE = self._generate_response(question)
        return self.RESPONSE

    def _generate_response(self, question: str) -> str:
        temp = ".".join(self.CONTEXT)
        memory_content = ""
        if self.memory is not None:
            memory_content = self.memory.get_memory()

        template = f"""
        {self.system_prompt}

        CONTEXT: {temp}
        --------------------
        CHAT HISTORY: {memory_content}
        QUERY: {question}
        """

        response = self.llm.predict(template)
        
        if self.memory is not None:
            self.memory.add_to_memory(question=question, response=response)
        
        return response

    def get_rag_triad_evals(self, llm = None):
        if llm is None:
            llm = self.llm
        print("Executing RAG Triad Evaluations...")
        context_relevancy = self.get_context_relevancy(llm)
        answer_relevancy = self.get_answer_relevancy(llm)
        groundness = self.get_groundedness(llm)
        return f"{context_relevancy}\n{answer_relevancy}\n{groundness}"
        
    def get_context_relevancy(self, llm = None):
        if llm is None:
            llm = self.llm
            
        total_score = 0
        score_count = 0
        
        for context in self.CONTEXT:
            score_str = llm.predict(CONTEXT_RELEVENCE.format(question=self.question, context=context))
            score = float(extract_number(score_str))
            total_score += score
            score_count += 1

        if score_count > 0:
            average_score = total_score / score_count
        else:
            average_score = 0

        return f"Context relevancy Score: {round(average_score, 1)}\n{thresholdCheck(average_score)}"

    def get_answer_relevancy(self, llm = None):
        if llm is None:
            llm = self.llm
        try:
            score_str = llm.predict(ANSWER_RELEVENCE.format(question=self.question, context=self.RESPONSE))
            score = float(extract_number(score_str))
    
            return f"Answer relevancy Score: {round(score, 1)}\n{thresholdCheck(score)}"
        except Exception as e:
            return(f"Failed during answer relevance evaluation: {e}")
        
    def get_groundedness(self, llm = None):
        if llm is None:
            llm = self.llm
        try:
            statements = sent_tokenize(self.RESPONSE)
            scores = []
            for statement in statements:
                score_response = llm.predict(GROUNDEDNESS.format(statement=statement, context=self.CONTEXT))
                score = extract_number(score_response)
                scores.append(score)
                
            average_score = sum(scores) / len(scores) if scores else 0
            
            return f"Groundness score: {round(average_score, 1)}\n{thresholdCheck(average_score)}"
        
        except Exception as e:
            raise Exception("Failed during groundedness evaluation: ", str(e))

    def get_ground_truth(self, answer:str, llm = None):
        if llm is None:
            llm = self.llm
        if answer is None:
            print("Error: Missing 'answer' parameter. Usage: get_ground_truth(answer=\"your_answer_here\").")
            return
        score_str = llm.predict(GROUND_TRUTH.format(ground_truth=answer, generated_response=self.RESPONSE))
        score = extract_number(score_str)
        
        return f"Ground truth score: {round(score, 1)}\n{thresholdCheck(score)}"


    @staticmethod
    def load_from_kwargs(self,kwargs): 
        model_config = GeneratorConfig(**kwargs)
        self.config = model_config
        self.pipeline()