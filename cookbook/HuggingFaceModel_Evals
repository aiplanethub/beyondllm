# pip install beyondllm
# pip install huggingface_hub
# pip install llama-index-embeddings-fastembed

from beyondllm.source import fit
from beyondllm.embeddings import FastEmbedEmbeddings
from beyondllm.retrieve import auto_retriever
from beyondllm.llms import HuggingFaceHubModel
from beyondllm.generator import Generate

import os
from getpass import getpass
os.environ['HUGGINGFACE_ACCESS_TOKEN'] = getpass("Enter your HF API token:")

data = fit("RedHenLab_GSoC_Tarun.pdf",dtype="pdf")
embed_model = FastEmbedEmbeddings()
retriever = auto_retriever(data=data,embed_model=embed_model,type="normal",top_k=3)
llm = HuggingFaceHubModel(model="mistralai/Mistral-7B-Instruct-v0.2")
pipeline = Generate(question="what models has Tarun fine-tuned?",llm=llm,retriever=retriever)

print(pipeline.call()) # Return the AI response
print(pipeline.get_rag_triad_evals())
