from enterprise_rag.llms import ChatOpenAIModel
#from enterprise_rag.utils import CONTEXT_RELEVENCE,GROUNDEDNESS,ANSWER_RELEVENCE

import os
from .base import BaseGenerator,GeneratorConfig
from dataclasses import dataclass,field

def default_llm():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY must be set in the environment variables.")
    return ChatOpenAIModel(api_key=api_key)

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
        context_relevancy = None
        answer_relevancy = None
        groundness = None

    def get_context_relevancy(self):
        return f"Context relevancy Score: {CONTEXT_RELEVENCE(self.CONTEXT,self.question)}"

    def get_answer_relevancy(self):
        return f"Answer relevancy Score: {ANSWER_RELEVENCE(self.RESPONSE,self.question)}"

    def get_groundness(self):
        return f"Groundness score: {GROUNDEDNESS(self.response,self.question)}"

    @staticmethod
    def load_from_kwargs(self,kwargs): 
        model_config = GeneratorConfig(**kwargs)
        self.config = model_config
        self.pipeline()