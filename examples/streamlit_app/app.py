import os
import streamlit as st
from beyondllm import generator

from beyondllm.llms import GeminiModel
from beyondllm import generator
from ingest import get_retriever

st.title("Chat with CSV file")

st.text("Enter Google API Key")
google_api_key = st.text_input("Google API Key:", type="password")
os.environ['GOOGLE_API_KEY'] = google_api_key

vectordb_options = ['Chroma', 'Pinecone']
with st.sidebar:
    st.title("VectorDB Options")
    vectordb_type = st.selectbox("Select VectorDB Type", 
                                 vectordb_options, 
                                 index=0)

    if vectordb_type == 'Pinecone':
        # get the pinecone api key and index name
        pinecone_api_key = st.text_input("Pinecone API Key:", 
                                         type="password")
        pinecone_index_name = st.text_input("Pinecone Index Name:")
        
        # choose whether to use an existing index or create a new one
        st.subheader("Pinecone Options")
        pinecone_option = st.radio("Choose Option", 
                                   ('Existing', 'Create New'), 
                                   index=0)

if google_api_key:
    st.success("Google API Key entered successfully!")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        st.success("file uploaded successfully!")
    question = st.text_input("Enter your question")

    if uploaded_file is not None and question:
        # Get the retriever
        if vectordb_type == 'Pinecone':
            retriever = get_retriever(uploaded_file, 
                                      google_api_key, 
                                      vector_db=vectordb_type.lower(), 
                                      pinecone_api_key=pinecone_api_key, 
                                      pinecone_index_name=pinecone_index_name, 
                                      pinecone_option=pinecone_option)
        elif vectordb_type == 'Chroma':
            retriever = get_retriever(uploaded_file, 
                                      google_api_key, 
                                      vector_db=vectordb_type.lower())
        # Initialize the LLM
        llm = GeminiModel(model_name="gemini-pro",
                          google_api_key = os.environ.get('GOOGLE_API_KEY'))
        # Initialize the system prompt
        system_prompt = "You are an AI assistant, who answers questions based on uploaded csv files. You can answer anything about the data."
        # Initialize the generator
        pipeline = generator.Generate(question=question,
                                      retriever=retriever, 
                                      llm=llm, 
                                      system_prompt=system_prompt)
        # Generate the response
        response = pipeline.call()
        # display the response
        st.write(response)

