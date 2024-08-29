import re
import streamlit as st
import validators
import time
from beyondllm import source, retrieve, embeddings, llms, generator
import os

# Dictionary to keep track of models skipped due to errors
skipped_due_to_err = {}

def colorize_multiselect_options(llm_to_model_tag: dict[str, tuple[str, ...]], colors: list[str]) -> None:
    """
    Applies custom background colors to multi-select options in a Streamlit interface based on the provided LLM and model tags.

    Parameters
    ----------
    llm_to_model_tag : dict[str, tuple[str, ...]])
        A dictionary where the keys represent LLM names (e.g., "OpenAI") and the values are tuples containing model tags (e.g., "gpt-3.5-turbo", "gpt-4").
        
    colors : list[str]
        A list of CSS-compatible color strings.
    
    Returns
    ----------
        None
    
    Example
    ----------
        >>> llm_to_model_tag = {"OpenAI": ("gpt-4", "gpt-4-turbo")}
        >>> colors = ["darkcyan"]
        >>> colorize_multiselect_options(llm_to_model_tag, colors)
    """
    rules = ""
    llms = list(llm_to_model_tag.keys())

    for i, color in enumerate(colors):
        llm = llms[i]
        rules += f""".stMultiSelect div[data-baseweb="select"] span[aria-label="{llm}, close by backspace"]{{background-color: {color};}}"""
        for m in llm_to_model_tag[llm]:
            rules += f""".stMultiSelect div[data-baseweb="select"] span[aria-label="{m}, close by backspace"]{{background-color: {color};}}"""

    st.markdown(f"<style>{rules}</style>", unsafe_allow_html=True)

def clean_and_validate(config):
    """
    Cleans and validates the configuration dictionary to remove invalid or incomplete entries.

    Parameters
    ----------
    config : dict
        Configuration dictionary where keys are LLM names and values are configurations for those LLMs.

    Returns
    ----------
    dict
        A dictionary with two keys: 'unvalid', a list of invalid LLM configurations, and 'cleaned_config', a dictionary of valid configurations.
    
    Example
    ----------
        >>> multiple_llm_configs = {"OpenAI": {"models": ["gpt-4", "gpt-4-turbo"], "api_key": "..."}}
        >>> clean_and_validate(multiple_llm_configs)
    """

    cleaned_config = {}
    unvalid=[]
    
    for key, value in config.items():
        # Remove models if empty
        if 'models' in value and not value['models']:
            break

        # Determine required fields dynamically (all keys apart from 'models')
        required_fields = [k for k in value.keys() if k != 'models']

        # Validate required fields are not empty or None
        valid = True
        for field in required_fields:
            if not value[field]:
                valid = False
                break
        if valid:
            cleaned_config[key] = value
        else:
            unvalid.append(key)

    return {"unvalid": unvalid, "cleaned_config": cleaned_config}

def pipeline(data, embed_model, llm_selected, api_key, model_name, question, azure_oai_endpoint=None):
    """
    Executes a query pipeline involving a data retriever, an embedding model, and a selected LLM.

    Parameters
    ----------
    data : str
        The data to be used for retrieval and embedding.
    embed_model : object
        The model used for embedding.
    llm_selected : str
        The name of the LLM to be used.
    api_key : str
        API key for the LLM service.
    model_name : str
        The specific model within the LLM service to use.
    question : str
        The question to ask the LLM.
    azure_oai_endpoint : str, optional
        The endpoint URL for Azure OpenAI (if applicable).

    Returns
    ----------
    dict or None
        A dictionary with the response and evaluation metrics, or None if an error occurs.
    """

    try:
        # Initialize the LLM model based on selection
        if llm_selected == "Gemini":
            llm_model = llms.GeminiModel(model_name=model_name, google_api_key=api_key)
        elif llm_selected == "OpenAI":
            llm_model = llms.ChatOpenAIModel(model=model_name, api_key=api_key)
        elif llm_selected == "Azure OpenAI":
            llm_model = llms.AzureOpenAIModel(model=model_name, azure_key = api_key, deployment_name=model_name, endpoint_url=azure_oai_endpoint)
        elif llm_selected == "Anthropic":
            llm_model = llms.ClaudeModel(api_key=api_key, model=model_name)
        elif llm_selected == "Groq":
            llm_model = llms.GroqModel(groq_api_key=api_key, model=model_name)
        elif llm_selected == "Hugging Face":
            llm_model = llms.HuggingFaceHubModel(token=api_key, model=model_name)

        # Initialize the data retriever
        retriever = retrieve.auto_retriever(data, type="normal", top_k=3, embed_model=embed_model)

        # Create and run the pipeline
        pipeline = generator.Generate(question=question, retriever=retriever, llm=llm_model)
        final_rag_output = pipeline.call()

        try:
            # Retrieving & formatting the evaluation scores
            str_evals = str(pipeline.get_rag_triad_evals()).lower()
            evaluation_result = {
                "context_relevancy": float(str_evals.split("context relevancy score: ")[1].split("\n")[0]),
                "answer_relevancy": float(str_evals.split("answer relevancy score: ")[1].split("\n")[0]),
                "groundedness": float(str_evals.split("groundness score: ")[1].split("\n")[0]),
            }
        except Exception as err:
            evaluation_result = {
                "context_relevancy": "-",
                "answer_relevancy": "-",
                "groundedness": "-"
            }
        return {
            "response": final_rag_output,
            "evaluation_metrics": evaluation_result
        }
    except Exception as err:
        global skipped_due_to_err
        if(skipped_due_to_err.get(llm_selected)):
            skipped_due_to_err[llm_selected].append({"model": model_name, "err": str(err)})
        else:
            skipped_due_to_err[llm_selected] = [{"model": model_name, "err": str(err)}]
        return None

def color_for_value(value):
    """
    Determines the color representation for a given evaluation metric value.

    Parameters
    ----------
    value : float
        The evaluation metric value to be colored.

    Returns
    ----------
    str
        The color associated with the given value.
    """

    return "red" if ((str(value) == '-') or (value <= 3)) else "yellow" if value <= 6 else "green" if value > 6 else "red"

# Mapping LLMs to available models
llm_to_model_tag = {
    "Gemini": ("gemini-1.0-pro", "gemini-pro", "gemini-1.5-pro-latest"),
    "OpenAI": ("gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-turbo"),
    "Azure OpenAI": ("gpt-35-turbo", "gpt-4", "gpt-35-turbo-16k"),
    "Anthropic": ("claude-3-5-sonnet-20240620", "claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"),
    "Groq": ("mixtral-8x7b-32768", "gemma2-9b-it", "llama-3.1-8b-instant", "llama3-70b-8192", "llama3-8b-8192", "llama3-groq-70b-8192-tool-use-preview", "llama3-groq-8b-8192-tool-use-preview"),
    "Hugging Face": ("huggingfaceh4/zephyr-7b-alpha", "huggingfaceh4/zephyr-7b-beta")
}

# Sidebar configuration section
st.sidebar.title("Configure:")

# LLM selection dropdown
selected_llms = st.sidebar.multiselect("Select your LLMs:", list(llm_to_model_tag.keys()))

# Apply custom colors to multi-select options
colors = ["darkcyan", "LightSlateGray", "RosyBrown", "tan", "mediumpurple", "SteelBlue"]
colorize_multiselect_options(llm_to_model_tag, colors)

# Initialize configuration dictionary for multiple LLMs
multiple_llm_configurations = {}
for llm in selected_llms:
    multiple_llm_configurations[llm] = {"api_key": None, "models": []}
    if(llm == "Azure OpenAI"):
        multiple_llm_configurations[llm]["oai_embeddings_model_name"] = None
        multiple_llm_configurations[llm]["azure_oai_endpoint"] = None
        
# Update configuration based on selected models and LLMs
if(selected_llms):
    models = st.sidebar.multiselect("Select your corresponding models:", list(set([item for row in [llm_to_model_tag[x] for x in selected_llms] for item in row])))
    if(set(["OpenAI", "Azure OpenAI"]).issubset(set(selected_llms))):
        st.sidebar.warning("Note: If you select the 'gpt-4' model, it will be applied to both Azure OpenAI and OpenAI LLMs for evaluation purposes.")
    
    # Display and update the configurations for each selected LLM
    for llm in selected_llms:
        multiple_llm_configurations[llm]["api_key"] = st.sidebar.text_input(label=f"Provide your API Key for {llm}:", type="password")
        if(llm == "Azure OpenAI"):
            azure_oai_endpoint = st.sidebar.text_input(label="What's your Azure OpenAI Endpoint URL?", placeholder="https://<azure-openai-resource-name>.openai.azure.com/")
            # URL validation check
            if (azure_oai_endpoint.strip() != "") and (not bool(re.match("https:\/\/(.+)\.openai\.azure\.com[/]{0,1}", azure_oai_endpoint))):
                st.sidebar.error("Please enter a valid endpoint!")
            elif(azure_oai_endpoint.strip()):
                multiple_llm_configurations[llm]["azure_oai_endpoint"] = azure_oai_endpoint
            multiple_llm_configurations[llm]["oai_embeddings_model_name"] = st.sidebar.text_input(label="Embeddings model name?", value="text-embedding-ada-002")
        for m in llm_to_model_tag[llm]:
            if(m in models):
                multiple_llm_configurations[llm]["models"].append(m)

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
st.title("Perform Multiple LLM Evaluations SimultaneouslyⓂ️")

# Clean and validate the LLM configurations
cleaned_llm_configurations = clean_and_validate(multiple_llm_configurations)

# Check if all required configurations are set
disabled = True
if(selected_llms != [] and models != [] and cleaned_llm_configurations["cleaned_config"] != {}):
    disabled = False

if(disabled):
    st.error("Make sure your have done all the configrations before querying!")

# Question input
question = st.text_input(
    label=f"Ask LLMs a question from {'your youtube video' if(source_selected.lower() == 'youtube') else 'your document'}:",
    value=(
        st.secrets["DEFAULTS"]["PAGE_QUERY"]
        if (source_selected.lower() == "url")
        else (st.secrets["DEFAULTS"]["YOUTUBE_QUERY"] if (source_selected.lower() == "youtube") else "")
    ),
    disabled=disabled,
)

# Run query button disabled based on input validation or missing credentials or missing question
to_run = st.button("Run query", disabled=(disabled or (question.strip() == "")))

# Display a warning if any selected models are not fully configured
if(not disabled and (cleaned_llm_configurations['unvalid'])):
    st.warning(f"Warning: {cleaned_llm_configurations['unvalid']} model(s) are selected but not fully configured, please review before proceeding.")

msg = None
try:
    if question.strip() and to_run:
        final_response = {}
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

        # Load and process the source data
        data = source.fit(url, dtype=str(source_selected).lower(), chunk_size=512, chunk_overlap=0)
        msg.toast("External data loaded!", icon="✅"); time.sleep(1)
        msg.toast("Configuring embedding models, generating responses by leveraging your selected LLM models... Please wait...", icon="🏗️"); time.sleep(5)

        # Initialize progress trackings
        total_iterations = sum(len(model["models"]) for model in cleaned_llm_configurations["cleaned_config"].values())
        current_iteration = 0

        progress_placeholder = st.empty()
        progress_bar = st.progress(0)

        # Iterate through each LLM and model for processing
        for llm_selected, model in cleaned_llm_configurations["cleaned_config"].items():
            if llm_selected == "Gemini":
                # uses 'models/embedding-001' by default for embeddings
                embed_model = embeddings.GeminiEmbeddings(api_key=model["api_key"])
            elif llm_selected == "OpenAI":
                # uses 'text-embedding-3-small' by default for embeddings
                embed_model = embeddings.OpenAIEmbeddings(api_key=model["api_key"])
            elif llm_selected == "Azure OpenAI":
                embed_model = embeddings.AzureAIEmbeddings(deployment_name=model["oai_embeddings_model_name"], azure_key=model["api_key"], endpoint_url=model["azure_oai_endpoint"])
            elif llm_selected == "Anthropic":
                # uses 'BAAI/bge-small-en-v1.5' by default for embeddings
                embed_model = embeddings.FastEmbedEmbeddings()
            elif llm_selected == "Groq":
                # uses 'BAAI/bge-small-en-v1.5' by default for embeddings
                embed_model = embeddings.FastEmbedEmbeddings()
            elif llm_selected == "Hugging Face":
                # uses 'BAAI/bge-small-en-v1.5' by default for embeddings
                # embed_model = embeddings.HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                embed_model = embeddings.FastEmbedEmbeddings()

            final_response[llm_selected] = []
            
            for m in model["models"]:
                # Update progress indicators
                progress_percentage = current_iteration / total_iterations
                progress_placeholder.text(f"Processing LLM: {llm_selected}, Model: {m} ({int(progress_percentage * 100)}% completed)")
                progress_bar.progress(progress_percentage)

                # Run the RAG and evaluation pipeline
                result = pipeline(data=data, embed_model=embed_model, llm_selected=llm_selected, api_key=model["api_key"], model_name=m, question=question, azure_oai_endpoint= model["azure_oai_endpoint"] if model.get("azure_oai_endpoint") else None)
                if(result):
                    final_response[llm_selected].append({"model": m, "result": result})
                
                # Update progress
                current_iteration += 1
                progress_percentage = current_iteration / total_iterations

                progress_placeholder.text(f"Processed LLM: {llm_selected}, Model: {m} ({int(progress_percentage * 100)}% completed)")
                progress_bar.progress(progress_percentage)
        
        # Clear progress indicators
        progress_placeholder.empty()
        progress_bar.empty()

        # Filter responses to include only valid results
        filtered_responses_and_evaluation_metrics = {llm: [item for item in model if item['result'] is not None] for llm, model in final_response.items() if [item for item in model if item['result'] is not None]}
        
        if(filtered_responses_and_evaluation_metrics):
            # Generate HTML table for displaying results
            html_string = """
                <body>
                <style>
                    table {
                        border-collapse: collapse;
                        width: 100%;
                    }

                    td, th {
                        border: 1px solid #dddddd;
                        text-align: left;
                        padding: 8px;
                    }
                </style>
            """ + f"""
                <h4>Question:</h4>
                    <p>{question}</p>
                    <br />
                    <table style="width:100%">
                        <tr>
                            <th style="text-align: left;">LLM</th>
                            <th style="text-align: left;">Model</th>
                            <th style="text-align: left;">Model Response</th>
                            <th style="text-align: left;">Context Relevancy</th>
                            <th style="text-align: left;">Answer Relevancy</th>
                            <th style="text-align: left;">Groundedness</th>
                        </tr>
            """

            # Add rows for each model and evaluation metrics
            for llm, model_details in filtered_responses_and_evaluation_metrics.items():
                for m in model_details:
                    html_string += f"""
                            <tr>
                                <td>{llm}</td>
                                <td>{m['model']}</td>
                                <td>{m['result']['response']}</td>
                                <td><div style="border: 1px solid {color_for_value(m['result']['evaluation_metrics']['context_relevancy'])}; padding: 4px; margin-right: 7px; background-color: {color_for_value(m['result']['evaluation_metrics']['context_relevancy'])}; border-radius: 4px; color: black; text-align: center;">{m['result']['evaluation_metrics']['context_relevancy']}</div></td>
                                <td><div style="border: 1px solid {color_for_value(m['result']['evaluation_metrics']['answer_relevancy'])}; padding: 4px; margin-right: 7px; background-color: {color_for_value(m['result']['evaluation_metrics']['answer_relevancy'])}; border-radius: 4px; color: black; text-align: center;">{m['result']['evaluation_metrics']['answer_relevancy']}</div></td>
                                <td><div style="border: 1px solid {color_for_value(m['result']['evaluation_metrics']['groundedness'])}; padding: 4px; background-color: {color_for_value(m['result']['evaluation_metrics']['groundedness'])}; border-radius: 4px; color: black; text-align: center;">{m['result']['evaluation_metrics']['groundedness']}</div></td>
                            </tr>
                        """
            
            st.html(html_string)

            st.caption(":orange[**Disclaimer:** These metrics are generated with the use of the same LLM and the detailed prompts defined by the team of beyondllm. Hence, the response may not be fully accurate. Please use your judgment and consult original source for verification.]")

        # Display completion message and error logs if any models failed        
        err_message = f" Some of the models from {str(list(skipped_due_to_err.keys()))} LLM(s) failed and were skipped! Check out the error logs for more details.." if skipped_due_to_err else ""
        new_toast = st.toast(f"Response with Evaluation executed successfully!{err_message}", icon="🥳"); time.sleep(5)

        # If there are any errors that occurred during model processing, display an expandable section in the Streamlit app to show the error logs.
        if(skipped_due_to_err):
            expander = st.expander("Error logs:")
            expander.write(skipped_due_to_err)
except Exception as err:
    # Catch any unexpected exceptions that occur during execution.
    st.write(err)
    st.toast("Uh-ohh! Something went wrong...", icon="☹️")
    time.sleep(1)