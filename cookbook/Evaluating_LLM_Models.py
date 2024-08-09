# This script requires the following external libraries:
# - beyondllm
# - validators
# - llama-index-readers-web
# - youtube_transcript_api
# - llama-index-readers-youtube-transcript
# - llama-index-embeddings-azure_openai
# - anthropic
# - llama-index-embeddings-fastembed
# - docx2txt

# Install them using pip:
#   pip install beyondllm validators llama-index-readers-web youtube_transcript_api llama-index-readers-youtube-transcript llama-index-embeddings-azure_openai anthropic llama-index-embeddings-fastembed docx2txt

import os
import re
import time
import validators
import streamlit as st
import streamlit.components.v1 as components
from beyondllm import source, retrieve, embeddings, llms, generator


def format_rag_triad_evals(evals: str) -> dict:
    """
    Parses the raw RAG triad evaluation string into a well-structured dictionary 
    containing context_relevancy, answer_relevancy, and groundness scores.

    Parameters
    ----------
        evals (str): The raw evaluation string output by the pipeline.

    Returns
    ----------
        dict: A dictionary containing the formatted evaluation scores.
    """
    lower_evals = str(evals).lower()
    result = {
        "context_relevancy": float(lower_evals.split("context relevancy score: ")[1].split("\n")[0]),
        "answer_relevancy": float(lower_evals.split("answer relevancy score: ")[1].split("\n")[0]),
        "groundness": float(lower_evals.split("groundness score: ")[1].split("\n")[0]),
    }
    return result


def metric_custom_css() -> None:
    """
    Modifying CSS of evaluation metrics displayed in st.metric based on their dynamic values.
    Injects the CSS code into the Streamlit app using the components.html function.

    Parameters
    ----------
        None
    Returns
    ----------
        None
    """
    
    html_string = """
        <script>
            const metricLabels = window.parent.document.querySelectorAll("[data-testid='stMetricLabel']");
            for (const metricLabel of metricLabels)
            {
                metricLabel.style.width = "fit-content";
                metricLabel.style.margin = "auto";
            }

            const metricValues = window.parent.document.querySelectorAll("[data-testid='stMetricValue']");
            for (const metricValue of metricValues)
            {
                const value = parseFloat(metricValue.textContent);
                let color = ((value <= 3) ? "red" : ((value <= 6) ? "yellow" : "green"));
                metricValue.style.color = color;
                metricValue.style.textAlign = "center";
            }
        </script>
        """
    components.html(f"{html_string}")

# Mapping LLMs to available models
llm_to_model_tag = {
    "Gemini": ("gemini-1.0-pro", "gemini-pro", "gemini-1.5-pro-latest"),
    "OpenAI": ("gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"),
    "Azure OpenAI": ("gpt-35-turbo", "gpt-4", "gpt-35-turbo-16k"),
    "Anthropic": ("claude-3-5-sonnet-20240620", "claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229")
}

# Sidebar configuration section
st.sidebar.title("Configure:")

# LLM selection dropdown
llm_selected = st.sidebar.selectbox(label="Which LLM you want to use?", options=llm_to_model_tag.keys())

if(llm_selected=="Azure OpenAI"):
    azure_oai_endpoint = st.sidebar.text_input(label="What's your Azure OpenAI Endpoint URL?", placeholder="https://<azure-openai-resource-name>.openai.azure.com/")
    
    # URL validation check
    if (azure_oai_endpoint.strip() != "") and (not bool(re.match("https:\/\/(.+)\.openai\.azure\.com[/]{0,1}", azure_oai_endpoint))):
        st.sidebar.error("Please enter a valid Endpoint!")

# API key input for selected LLM
api_key = st.sidebar.text_input(label=f"Provide your API Key{(' for '+ str(llm_selected)) if llm_selected else ''}:", type="password")

# Model selection dropdown based on chosen LLM
model_name = st.sidebar.selectbox(label=f"Which model from {llm_selected} you want to use?", options=llm_to_model_tag[llm_selected])

if(llm_selected == "Azure OpenAI"):
    # Configuration for Azure OpenAI LLM
    azure_oai_deployment_name = st.sidebar.text_input(label=f"Deployment name for {model_name}?", value=model_name)
    embeddings_oai_model_name = st.sidebar.text_input(label="Embeddings model name?", value="text-embedding-ada-002")
elif(llm_selected == "Anthropic"):
    # Information for Anthropic LLM
    st.sidebar.warning("We use FastEmbedEmbedding('BAAI/bge-small-en-v1.5') model from llama_index for embeddings!")

# Warning message if no models available for chosen LLM
if llm_to_model_tag[llm_selected] == ():
    st.sidebar.warning("Uh-ohh! Models are getting added soon..")

# Source selection and content for RAG in columns layout
source_column, second_column = st.sidebar.columns(2)
source_selected = source_column.selectbox(label="Which source?", options=("URL", "YouTube", "PDF", "DOCX"))

if(source_selected in ("URL", "YouTube")):
    # Prompt the user for the content link in the second column based on user selection
    url = second_column.text_input(
        label="Link to your content:",
        value=(
            st.secrets["DEFAULTS"]["PAGE_URL"]
            if (source_selected.lower() == "url")
            else st.secrets["DEFAULTS"]["YOUTUBE_URL"]
        ),
    )

    # URL validation check
    if (url.strip() != "") and (not validators.url(url)):
        st.sidebar.error("Please enter a valid URL!")
else:
    # If not URL or YouTube, allow file upload based on selected source
    uploaded_file = st.sidebar.file_uploader(f"Choose a {source_selected.upper()} file", type=source_selected.lower())

# Main content section
st.title("Perform quick LLM evaulations🔄️")

# Custom CSS for better formatting
st.markdown(
    """
        <style>
            .block-container { padding-top: 2.5rem; padding-bottom: 1rem; padding-left: 1rem; padding-right: 1rem; }
            span[data-testid="stHeaderActionElements"] { display:none; }
            [data-testid="stMetricLabel"] > div { text-align: center; }
            iframe { display: none; }
            .element-container > .stMarkdown > div[data-testid="stCaptionContainer"] { text-align: center; }
        </style>
    """,
    unsafe_allow_html=True,
)

# Button disabled state based on input validation and missing credentials
disabled = (
        api_key.strip() != "" and
        ((bool(re.match("https:\/\/(.+)\.openai\.azure\.com[/]{0,1}", azure_oai_endpoint)) and (str(azure_oai_deployment_name).strip() != "") and (str(embeddings_oai_model_name).strip() != "")) if llm_selected=="Azure OpenAI" else True)
    )
try:
    disabled = not (disabled and ((url.strip() != "") and (validators.url(url))))
except NameError:
    disabled = not (disabled and (uploaded_file is not None))
if not disabled:
    os.environ["API_KEY"] = api_key
else:
    st.error("Make sure your have done all the configurations before querying!")

# Question input with dynamic label based on source selection
question = st.text_input(
    label=f"Ask {llm_selected} a question from {'your youtube video' if(source_selected.lower() == 'youtube') else 'your document'}:",
    value=(
        st.secrets["DEFAULTS"]["PAGE_QUERY"]
        if (source_selected.lower() == "url")
        else (st.secrets["DEFAULTS"]["YOUTUBE_QUERY"] if (source_selected.lower() == "youtube") else "")
    ),
    disabled=disabled,
)

# Run query button disabled based on input validation or missing credentials or missing question
to_run = st.button("Run query", disabled=(disabled or (question.strip() == "")))

try:
    if question.strip() and to_run:
        # ---------------------- EXTERNAL DATA LOAD ----------------------
        msg = st.toast("Loading external data...", icon="⚙️"); time.sleep(1)

        if(source_selected.lower() in ("pdf", "docx")):
            # Define the path to save uploaded files
            save_path = "./uploaded_files"

            # Create the directory if it doesn't exist
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Combine the save path and uploaded file name to create the complete URL
            url = os.path.join(save_path, uploaded_file.name)

            # Open the file in binary write mode for safe storage of various file types
            with open(url, "wb") as f:
                f.write(uploaded_file.getbuffer())

        data = source.fit(url, dtype=str(source_selected).lower(), chunk_size=512, chunk_overlap=0)

        msg.toast("External data loaded!", icon="✅"); time.sleep(1)
        msg.toast("Configuring embedding and LLM model...", icon="🔢"); time.sleep(1)
        
        # ---------------------- LLM MODEL CONFIGURATION ----------------------
        # LLM model selection and configuration based on chosen provider
        if llm_selected == "Gemini":
            # uses 'models/embedding-001' by default for embeddings
            embed_model = embeddings.GeminiEmbeddings(api_key=os.environ["API_KEY"])
            llm_model = llms.GeminiModel(model_name=model_name, google_api_key=os.environ["API_KEY"])
        elif llm_selected == "OpenAI":
            # uses 'text-embedding-3-small' by default for embeddings
            embed_model = embeddings.OpenAIEmbeddings(api_key=os.environ["API_KEY"])
            llm_model = llms.ChatOpenAIModel(model=model_name, api_key=os.environ["API_KEY"])
        elif llm_selected == "Azure OpenAI":
            embed_model = embeddings.AzureAIEmbeddings(deployment_name=embeddings_oai_model_name, azure_key=os.environ["API_KEY"], endpoint_url=azure_oai_endpoint)
            llm_model = llms.AzureOpenAIModel(model=model_name, azure_key = os.environ["API_KEY"], deployment_name=azure_oai_deployment_name, endpoint_url=azure_oai_endpoint)
        elif llm_selected == "Anthropic":
            # uses 'BAAI/bge-small-en-v1.5' by default for embeddings
            embed_model = embeddings.FastEmbedEmbeddings()
            llm_model = llms.ClaudeModel(api_key=os.environ["API_KEY"], model=model_name)
        
        msg.toast("Embedding and LLM model configured!", icon="👍"); time.sleep(1)

        # ---------------------- RETRIEVE DOCUMENTS AND RUNNING GENERATOR MODEL ----------------------
        msg.toast("Generating response by leveraging retriever and LLM...", icon="🏗️"); time.sleep(1)
        
        retriever = retrieve.auto_retriever(data, type="normal", top_k=3, embed_model=embed_model)
        pipeline = generator.Generate(question=question, retriever=retriever, llm=llm_model)
        
        final_rag_output = pipeline.call()
        
        # ---------------------- SUMMARIZED RESPONSE DISPLAY IN CONTAINER ----------------------
        st.container(border=True).write(f"**Response:**  \n   \n {final_rag_output}")
        
        msg.toast("Response generated!", icon="🥳"); time.sleep(1)

        # ---------------------- EVALUATION SCORES ----------------------
        msg.toast("Assessing response on evaluation benchmarks...", icon="📊"); time.sleep(1)
        
        cr_col, ar_col, grnd_col = st.columns(3)
        # Format the retrieved evaluation scores
        formatted_evals = format_rag_triad_evals(pipeline.get_rag_triad_evals())

        # Displaying evaluation scores as metric
        cr_col.metric(label="**Context Relevancy**", value=f"{formatted_evals['context_relevancy']}", help="This score between 0 (least relevant) to 10 (most relevant) indicates how well the system finds information related to your query.")
        ar_col.metric(label="**Answer Relevance**", value=f"{formatted_evals['answer_relevancy']}", help="This score between 0 (completely irrelevant) to 10 (highly relevant) indicates how well the LLM's response aligns with your question.")
        grnd_col.metric(label="**Groundedness**", value=f"{formatted_evals['groundness']}", help="This score between 0 (completely hallucinated) to 10 (fully grounded) determines the extent to which the LLM's responses are grounded.")

        if(formatted_evals):
            st.caption(":orange[**Disclaimer:** These metrics are generated with the use of the same LLM and the detailed prompts defined by the team of beyondllm. Hence, the response may not be fully accurate. Please use your judgment and consult original source for verification.]")

        # Applying custom CSS for formatting metrics
        metric_custom_css()

        msg.toast("Evaluation completed successfully!", icon="🖨️"); time.sleep(1)

except Exception as err:
    # Error handling for API key related issues for supported models
    if ("400 API key not valid." in err.__str__()) or ("Incorrect API key provided" in err.__str__()):
        st.error("Check your credentials!")
    else:
        st.write("An error occurred:", err)

    msg.toast("Uh-ohh! Something went wrong...", icon="☹️"); time.sleep(1)
