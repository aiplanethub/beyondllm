import os
from beyondllm.retrieve import auto_retriever
from beyondllm.vectordb import ChromaVectorDb, PineconeVectorDb, WeaviateVectorDb, RedisVectorDb
from beyondllm.embeddings import GeminiEmbeddings
from beyondllm import source

def get_retriever(uploaded_file, google_api_key, vector_db='chroma', 
                  pinecone_api_key=None, pinecone_index_name=None, 
                  pinecone_option=None, pinecone_embedding_dim=None, 
                  pinecone_metric=None, pinecone_cloud=None, 
                  pinecone_region=None, weaviate_url=None, 
                  weaviate_index_name=None, weaviate_api_key=None, 
                  weaviate_headers=None, redis_host=None, 
                  redis_port=None, redis_index_name=None, redis_password=None):
    if google_api_key:
        # Save the uploaded file
        save_path = "./uploaded_files" # change this to your desired path or leave it as is
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Fit the data
        data = source.fit(file_path, dtype="csv", 
                          chunk_size=512, 
                          chunk_overlap=50)
        # Initialize your embedding model
        embed_model = GeminiEmbeddings(api_key=google_api_key, 
                                       model_name="models/embedding-001")
        # Initialize your vector store
        if vector_db == 'chroma':
            vector_store = ChromaVectorDb(collection_name='my_persistent_collection', # change this to your desired collection name
                                          persist_directory='./db/chroma/')
        elif vector_db == 'pinecone':
            if pinecone_option == 'Existing':
                # Initialize an existing Pinecone index
                vector_store = PineconeVectorDb(api_key=pinecone_api_key, 
                                                index_name=pinecone_index_name)
            else:
                # Create a new serverless Pinecone index
                vector_store = PineconeVectorDb(
                    create=True,
                    api_key=pinecone_api_key,
                    index_name=pinecone_index_name,
                    embedding_dim=pinecone_embedding_dim,
                    metric=pinecone_metric,
                    cloud=pinecone_cloud,
                    region=pinecone_region,
                )
        elif vector_db == 'weaviate':
            vector_store = WeaviateVectorDb(
                url=weaviate_url,
                index_name=weaviate_index_name,
                api_key=weaviate_api_key,
                additional_headers=weaviate_headers
            )
        elif vector_db == 'redis':
            vector_store = RedisVectorDb(
                host=redis_host,
                port=redis_port,
                index_name=redis_index_name,
                password=redis_password
            )

        # Initialize the retriever
        retriever = auto_retriever(data=data, embed_model=embed_model, type="normal", top_k=5, vectordb=vector_store)
        
        return retriever

    return None
