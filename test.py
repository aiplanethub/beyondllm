from enterprise_rag.etl import fit
from enterprise_rag.embeddings import OpenAIEmbeddings
from enterprise_rag.retrieve import auto_retriever

# from llama_index.llms.azure_openai import AzureOpenAI


# data = fit(path="/home/adithya/Downloads/Example_doc.pdf",loader_type='llama-parse',chunk_size=512,chunk_overlap=50,llama_parse_key="llx-fONiYabJEOhMhJ3FOfiI9tGTHzrJRdkGnyzyO1UQFAjzsoxH")
data = fit(path="https://www.youtube.com/watch?v=N2Ul-KWJdHw",loader_type="youtube",chunk_size=100,chunk_overlap=50)
# print(data)

embed_model = OpenAIEmbeddings(api_key="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")

retriever = auto_retriever(data,embed_model,type="cross-rerank",top_k=7)

print(retriever.retrieve("Spinners")[0])


# embed_model = get_embed_model(
#     model_name="AzureOpenAI",
#     api_key="a20bc67dbd7c47ed8c978bbcfdacf930",
#     azure_endpoint="https://marketplace.openai.azure.com/",
#     deployment_name = "text-embed-marketplace",
#     api_version="2023-05-15",
#     )
# azure_embed = embed_model


# llm = AzureOpenAI(
#     model="gpt-35-turbo-16k",
#     deployment_name="gpt4-inference",
#     api_key="a20bc67dbd7c47ed8c978bbcfdacf930",
#     azure_endpoint="https://gpt-res.openai.azure.com/",
#     api_version="2023-07-01-preview",
# )

# print(type(retriever))

# results = retriever.evaluate(llm)
# print(results)