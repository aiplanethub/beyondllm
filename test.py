from enterprise_rag.etl import fit
from enterprise_rag.embed import get_embed_model
from enterprise_rag.retrieve import auto_retriever

from llama_index.llms.azure_openai import AzureOpenAI


<<<<<<< HEAD
# data = fit(path="/home/adithya/Downloads/Example_doc.pdf",loader_type='llama-parse',chunk_size=512,chunk_overlap=50,llama_parse_key="llx-fONiYabJEOhMhJ3FOfiI9tGTHzrJRdkGnyzyO1UQFAjzsoxH")
data = fit(path="/home/adithya/Downloads/LLavaPaper.pdf",chunk_size=100,chunk_overlap=50)
# print(data)
=======
# data = fit(path="/home/adithya/Downloads/Example_doc.pdf",loader_type='llama-parse',chunk_size=512,chunk_overlap=50,llama_parse_key="llx-*************")
data = fit(path="/home/adithya/Downloads/AdithyaHegde_Resume.pdf",chunk_size=512,chunk_overlap=50)
print(data)
print(type(data))
>>>>>>> 61a311a (Hide keys)

embed_model = get_embed_model(
    model_name="AzureOpenAI",
    api_key="***********",
    azure_endpoint="https://marketplace.openai.azure.com/",
    deployment_name = "text-embed-marketplace",
    api_version="2023-05-15",
    )
azure_embed = embed_model

retriever = auto_retriever(data,embed_model,type="normal")

print((retriever.retrieve("IIIT Dharwad")[0]))

llm = AzureOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name="gpt4-inference",
    api_key="a20bc67dbd7c47ed8c978bbcfdacf930",
    azure_endpoint="https://gpt-res.openai.azure.com/",
    api_version="2023-07-01-preview",
)

print(type(retriever))

<<<<<<< HEAD
results = retriever.evaluate(llm)
print(results)
=======
index = VectorStoreIndex(
    data, service_context=sentence_context
)

query_engine = index.as_query_engine()

response = query_engine.query("IIIT Dharwad")

print(response)










# os.environ["AZURE_OPENAI_API_KEY"] = "**************"
# os.environ["AZURE_OPENAI_ENDPOINT"] = "https://marketplace.openai.azure.com/"

# azure_embeddings = AzureOpenAIEmbedding(
#             api_key="**********",
#             azure_endpoint="https://marketplace.openai.azure.com/",
#             deployment_name = "text-embed-marketplace",
#             api_version="2023-05-15",
#         )
# llm = AzureOpenAI(
#     model="gpt-35-turbo-16k",
#     deployment_name="gpt4-inference",
#     api_key="**********",
#     azure_endpoint="https://gpt-res.openai.azure.com/",
#     api_version="2023-07-01-preview",
# )

# embed_models = {
#     "AzureOpenAI": azure_embeddings,
# }
# results = evaluate_embeddings(embed_models,data,llm)

# # print(len(data),"\n",embed_model)
# print(results)
# import os
# from llama_index.readers.web import SimpleWebPageReader
# from llama_index.core.node_parser import SentenceSplitter

# # docs = SimpleWebPageReader(html_to_text=True).load_data(["https://www.youtube.com/watch?v=ZLbVdvOoTKM"])  

# # splitter = SentenceSplitter(
# #     chunk_size=1024,
# #     chunk_overlap=200,
# # )
>>>>>>> 61a311a (Hide keys)
