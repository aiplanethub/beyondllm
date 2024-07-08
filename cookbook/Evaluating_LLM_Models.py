# This script requires the following external libraries:
# - beyondllm
# - validators
# - llama-index-readers-web
# - youtube_transcript_api
# - llama-index-readers-youtube-transcript

# Install them using pip:
#   pip install beyondllm validators llama-index-readers-web youtube_transcript_api llama-index-readers-youtube-transcript

import os
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
            const metricValues = window.parent.document.querySelectorAll("[data-testid='stMetricValue']");
            for (const metricValue of metricValues)
            {
                const value = parseFloat(metricValue.textContent);
                let color = ((value <= 3) ? "red" : ((value <=6) ? "yellow" : "green"));
                metricValue.style.color = color;
                metricValue.style.textAlign = "center";
            }
        </script>
        """
    components.html(f"{html_string}")

# Mapping LLMs to available models
llm_to_model_tag = {
    "Gemini": ("gemini-1.0-pro", "gemini-1.5-flash", "gemini-1.5-pro"),
    "OpenAI": ("gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"),
    "Azure OpenAI": (),
}

# Sidebar configuration section
st.sidebar.title("Configure:")

# LLM selection dropdown
llm_selected = st.sidebar.selectbox(label="Which LLM you want to use?", options=("Gemini", "OpenAI", "Azure OpenAI"))

# API key input for selected LLM
api_key = st.sidebar.text_input(label=f"Provide your API Key{(' for '+ str(llm_selected)) if llm_selected else ''}:", type="password")

# Model selection dropdown based on chosen LLM
model_name = st.sidebar.selectbox(label=f"Which model from {llm_selected} you want to use?", options=llm_to_model_tag[llm_selected])

# Warning message if no models available for chosen LLM
if llm_to_model_tag[llm_selected] == ():
    st.sidebar.warning("Uh-ohh! Models are getting added soon..")

# Source selection and content link for RAG in columns layout
source_column, url_column = st.sidebar.columns(2)
source_selected = source_column.selectbox(label="Which source?", options=("URL", "YouTube"))
url = url_column.text_input(
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

# Main content section
st.title("Perform quick model evaulations🔄️")

# Custom CSS for better formatting
st.markdown(
    """
        <style>
            span[data-testid="stHeaderActionElements"] { display:none; }
            [data-testid="stMetricLabel"] > div { text-align: center; }
        </style>
    """,
    unsafe_allow_html=True,
)

# Button disabled state based on input validation and missing credentials
disabled = not (api_key.strip() != "" and ((url.strip() != "") and (validators.url(url))))
if not disabled:
    os.environ["API_KEY"] = api_key
else:
    st.error("Make sure your have Model, API Key and URL set before querying!")

# Question input with dynamic label based on source selection
question = st.text_input(
    label=f"Ask {llm_selected} a question from {'your document' if(source_selected.lower() == 'url') else 'your youtube video'}:",
    value=(
        st.secrets["DEFAULTS"]["PAGE_QUERY"]
        if (source_selected.lower() == "url")
        else st.secrets["DEFAULTS"]["YOUTUBE_QUERY"]
    ),
    disabled=disabled,
)

# Run query button disabled based on input validation or missing credentials or missing question
to_run = st.button("Run query", disabled=(disabled or (question.strip() == "")))

try:
    if question.strip() and to_run:
        # ---------------------- EXTERNAL DATA LOAD ----------------------
        msg = st.toast("Loading external data...", icon="⚙️"); time.sleep(1)

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
        cr_col.metric(label="**Context Relevancy**", value=f"{formatted_evals['context_relevancy']}")
        ar_col.metric(label="**Answer Relevance**", value=f"{formatted_evals['answer_relevancy']}")
        grnd_col.metric(label="**Groundedness**", value=f"{formatted_evals['groundness']}")

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