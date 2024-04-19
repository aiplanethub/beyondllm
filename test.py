from beyondllm import source,retrieve,embeddings,llms,generator
from beyondllm.llms import GeminiModel
import os
os.environ['GOOGLE_API_KEY'] = "AIzaSyAsgNPR2-ve6QZqR9B5snRzW6Awm2-28o4"

data = source.fit(path="https://tarunjain.netlify.app", dtype="url",chunk_size=1024)
retriever = retrieve.auto_retriever(data,type="normal")
query = "who is he?"

print(retriever.retrieve(query))

pipeline = generator.Generate(question=query,retriever=retriever)
print(pipeline.call())
print(pipeline.get_rag_triad_evals())
