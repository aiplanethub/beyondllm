from beyondllm import source, retrieve, llms
from beyondllm.embeddings import FineTuneEmbeddings
import os


# Setting up an environment variable for API key
os.environ['GOOGLE_API_KEY'] = "your-api-key"

# Importing and preparing the data
data = source.fit("build-career-in-ai.pdf", dtype="pdf", chunk_size=1024, chunk_overlap=0)

# List of files to train the embeddings
list_of_files = ['build-career-in-ai.pdf']

# Initializing a Gemini LLM model
llm = llms.GeminiModel()

# Creating an instance of FineTuneEmbeddings
fine_tuned_model = FineTuneEmbeddings()

# Training the embedding model
embed_model = fine_tuned_model.train(list_of_files, "BAAI/bge-small-en-v1.5", llm, "fintune")

# Option to load an already fine-tuned model
# embed_model = fine_tuned_model.load_model("fintune")

# Creating a retriever using the fine-tuned embeddings
retriever = retrieve.auto_retriever(data, embed_model, type="normal", top_k=4)

# Retrieving information using a query
print(retriever.retrieve("How to excel in AI?"))
