from enterprise_rag.etl import fit
# from enterprise_rag.embed_data import get_embed_model
# # from enterprise_rag.embed_data import evaluate_embeddings
# from llama_index.llms.azure_openai import AzureOpenAI
# from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
# import os

# data = fit(path="/home/adithya/Downloads/Example_doc.pdf",loader_type='llama-parse',chunk_size=512,chunk_overlap=50,llama_parse_key="llx-fONiYabJEOhMhJ3FOfiI9tGTHzrJRdkGnyzyO1UQFAjzsoxH")
data = fit(path="/home/adithya/Downloads/AdithyaHegde_Resume.pdf",chunk_size=512,chunk_overlap=50)
print(data)






# # embed_model = get_embed_model(
# #     model_name="AzureOpenAI",
# #     azure_endpoint="https://marketplace.openai.azure.com/",
# #     azure_key="d6d9522a01c74836907af2f3fd72ff85",
# #     api_version="2023-05-15",
# #     azure_deployment="text-embed-marketplace"
# #     )
# # azure_embed = embed_model

# os.environ["AZURE_OPENAI_API_KEY"] = "d6d9522a01c74836907af2f3fd72ff85"
# os.environ["AZURE_OPENAI_ENDPOINT"] = "https://marketplace.openai.azure.com/"

# azure_embeddings = AzureOpenAIEmbedding(
#             api_key="d6d9522a01c74836907af2f3fd72ff85",
#             azure_endpoint="https://marketplace.openai.azure.com/",
#             deployment_name = "text-embed-marketplace",
#             api_version="2023-05-15",
#         )
# llm = AzureOpenAI(
#     model="gpt-35-turbo-16k",
#     deployment_name="gpt4-inference",
#     api_key="a20bc67dbd7c47ed8c978bbcfdacf930",
#     azure_endpoint="https://gpt-res.openai.azure.com/",
#     api_version="2023-07-01-preview",
# )

# embed_models = {
#     "AzureOpenAI": azure_embeddings,
# }
# results = evaluate_embeddings(embed_models,data,llm)

# # print(len(data),"\n",embed_model)
# print(results)