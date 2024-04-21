# pip install beyondllm
# pip install ollama
# pip install llama-index-embeddings-fastembed
from beyondllm import source,retrieve,generator
from beyondllm.llms import OllamaModel
from beyondllm.embeddings import FastEmbedEmbeddings

query = "what is Tarun's role at AI Planet?"
data = source.fit(path="https://tarunjain.netlify.app", dtype="url",chunk_size=1024)
retriever = retrieve.auto_retriever(data,embed_model=FastEmbedEmbeddings(),type="normal")
pipeline = generator.Generate(question=query,llm=OllamaModel(model="llama2"),retriever=retriever)

print(pipeline.call())
print(pipeline.get_rag_triad_evals())
