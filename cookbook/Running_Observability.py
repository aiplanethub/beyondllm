from beyondllm import source,retrieve,generator, llms, embeddings
from beyondllm.observe import Observer
import os

os.environ['OPENAI_API_KEY'] = 'sk-****'

Observe = Observer()
Observe.run()

llm=llms.ChatOpenAIModel()
embed_model = embeddings.OpenAIEmbeddings()

data = source.fit("https://www.youtube.com/watch?v=oJJyTztI_6g",dtype="youtube",chunk_size=512,chunk_overlap=50)
retriever = retrieve.auto_retriever(data,embed_model,type="normal",top_k=4)

pipeline = generator.Generate(question="what tool is video mentioning about?",retriever=retriever, llm=llm)
pipeline = generator.Generate(question="What is the tool used for?",retriever=retriever, llm=llm)
pipeline = generator.Generate(question="How can i use the tool for my own use?",retriever=retriever, llm=llm)


