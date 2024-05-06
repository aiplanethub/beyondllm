# 🌐 Source

## What is Source?

Source refers to the origin of the RAG pipeline: data. The first step in building a RAG pipeline is to source the data from diverse origins, and transform it for usability.  We follow a two step process: loading the data, followed by splitting/chunking of data.&#x20;

The beyondllm`.source`module provides a variety of loaders to ingest and process data from different sources. This allows you to easily integrate your data into the RAG pipeline for question answering and information retrieval. This returns a List of TextNode objects.

## fit Function

The central function for loading data is fit. It offers a unified interface for loading and processing data regardless of the source type.

```python
from beyondllm.source import fit

data = fit(path="<your-doc-path-here>", dtype="<your-dtype>", chunk_size=512, chunk_overlap=100)
```

**Parameters:**

* **path (str):** The path to the data source. This can be a local file path, a URL, or a YouTube video ID, depending on the chosen loader type.
* **dtype (str):** Specifies the type of loader to use based on the data source format or origin. Supported options include:
  * **File Types:** "pdf", "csv", "docx", "epub", "md", "ppt", "pptx", "pptm" (using SimpleLoader)
  * **Web Pages:** "url" (using UrlLoader)
  * **YouTube Videos:** "youtube" (using YoutubeLoader)
  * **LlamaParse Cloud API:** "llama-parse" (using LlamaParseLoader)
* **chunk\_size (int):** The desired size of each text chunk after splitting the document. Default is 512 characters.
* **chunk\_overlap (int):** The amount of overlap between consecutive chunks. This helps maintain context and coherence. Default is 100 characters.

**Returns:**

* **List\[TextNode]:** A list of TextNode objects representing the processed data, ready for further use in the RAG pipeline.

## Available Loaders

All of the below loaders are offered using the `fit` function itself.&#x20;

**NOTE:** The `dtype` parameter is the only value to be changed based on your input type and the `fit` function will do the job for you!&#x20;

The default chunk\_size is 512 and the default chunk\_overlap is 100.

### **SimpleLoader**

Handles common file types like PDFs, Word documents, presentations, and Markdown files. Supports chunk\_size and chunk\_overlap parameters for text splitting. This currently supports the following file formats: "pdf", "csv", "docx", "epub", "md", "ppt", "pptx", "pptm". However, the below command needs to be run to install the required library to parse .docx files:

```bash
pip install docx2text
```

For .ppt files:

```bash
pip install torch transformers python-pptx Pillow
```

Code snippet: Loading a PDF document:

```python
from beyondllm.source import fit

data = fit(path="path/to/document.pdf", dtype="pdf", chunk_size=1024, chunk_overlap=50)
```

### **UrlLoader**

Extracts text content from web pages based on the provided URL. Supports chunk\_size and chunk\_overlap parameters. This requires an additional library which can be installed by running:

```bash
pip install llama-index-readers-web
```

Code snippet: Loading HTML data from a website using the URL:

```python
from beyondllm.source import fit

data = fit(path="url-link-of-your-website.com", dtype="url")
```

### **YoutubeLoader**

Downloads and processes transcripts from YouTube videos. Supports chunk\_size and chunk\_overlap parameters.

```bash
pip install llama-index-readers-youtube-transcript youtube-transcript-api
```

Code snippet: Loading a youtube video transcript:

```python
from beyondllm.source import fit

data = fit(path="youtube-video-url", dtype="youtube")
```

### **LlamaParseLoader**

Leverages the LlamaParse Cloud API to extract structured information and text from various documents. Requires an additional llama\_parse\_key parameter or `LLAMA_CLOUD_API_KEY` environment variable to be set. Supports chunk\_size and chunk\_overlap parameters. We internally perform a markdown splitting on your data.

```bash
pip install llama-parse
```

**NOTE**: Currently only supports .pdf files, and requires a Llama Parse API key, you can obtain one from [https://cloud.llamaindex.ai/login](https://cloud.llamaindex.ai/login).

Code snippet: Loading a PDF document:

```python
from beyondllm.source import fit

data = fit(path="path/to/document.pdf", dtype="llama-parse", llama_parse_key="llx-",)
```

**NOTE:** If llama-parse loader is to be used in a Google Colab notebook, run the below lines of code before using fit with dtype="llama-parse"

```python
import nest_asyncio
nest_asyncio.apply()

data = fit(path="path/to/document.pdf", dtype="llama-parse", llama_parse_key="llx-",)
```

**By leveraging the diverse range of loaders in BeyondLLM, you can effectively incorporate information from various sources into your RAG pipeline for enhanced question answering and knowledge retrieval capabilities.**
