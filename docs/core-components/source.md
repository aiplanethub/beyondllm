# 🌐 Source

## What is Source?

Source refers to the origin of the RAG pipeline: data. The first step in building a RAG pipeline is to source the data from diverse origins, and transform it for usability.  We follow a two step process: loading the data, followed by splitting/chunking of data.&#x20;

The beyondllm`.source`module provides a variety of loaders to ingest and process data from different sources. This allows you to easily integrate your data into the RAG pipeline for question answering and information retrieval. This returns a List of TextNode objects.

## fit Function

The central function for loading data is fit. It offers a unified interface for loading and processing data regardless of the source type. Whether you have local files, web pages, YouTube videos, or want to leverage the power of LlamaParse, fit simplifies the process, handling multiple input types with ease.

**Centralized Data Loading:**

```python
from beyondllm.source import fit

data = fit(path="<your-doc-path-here>", dtype="<your-dtype>", chunk_size=512, chunk_overlap=100)
```

**Parameters:**

* **path** (str or list): The path to your data source(s).
  * For single inputs: A string representing a local file path, a URL, or a YouTube video ID.
  * For multiple inputs: A list of strings, where each string is a file path, URL, or YouTube video ID.
* **dtype** (str): Specifies the type of loader to use, based on your data format. Supported options include:
  * **File Types:** "pdf", "csv", "docx", "epub", "md", "ppt", "pptx", "pptm" (using SimpleLoader)
  * **Web Pages:** "url" (using UrlLoader)
  * **YouTube Videos:** "youtube" (using YoutubeLoader)
  * **LlamaParse Cloud API:** "llama-parse" (using LlamaParseLoader)
* **chunk\_size** (int): The desired length (in characters) for splitting text into chunks. Defaults to 512.
* **chunk\_overlap** (int): The number of overlapping characters between consecutive chunks to preserve context. Defaults to 100.

**Returns:**

* List\[TextNode]: A list of TextNode objects representing your processed data, ready for use in the RAG pipeline.

**Available Loaders:**

BeyondLLM provides a range of specialized loaders to handle different data types. All loaders are accessible through the fit function by simply changing the dtype parameter.

### **1. SimpleLoader:**

Handles common file types like PDFs, Word documents, presentations, and Markdown files. Supports chunk\_size and chunk\_overlap parameters for text splitting. This currently supports the following file formats: "pdf", "csv", "docx", "epub", "md", "ppt", "pptx", "pptm", "txt".&#x20;

* **File Types Supported:** "pdf", "csv", "docx", "epub", "md", "ppt", "pptx", "pptm", "txt"
* **Multiple Inputs:** Accepts a list of file paths.
* **Directory Loading:** When providing a directory path as path, the SimpleLoader will automatically load all supported file types within that directory.

**Code Snippet (Loading Multiple PDFs):**

```python
from beyondllm.source import fit

pdf_paths = ["path/to/document1.pdf", "path/to/document2.pdf", "path/to/document3.pdf"]
data = fit(path=pdf_paths, dtype="pdf")
```

or to load one file, it can just be passed as a path:

```python
from beyondllm.source import fit

data = fit(path="path/to/document1.pdf", dtype="pdf")
```

Some file types require some additional libraries. Install the required library to parse .docx files:

```bash
pip install docx2text
```

For .ppt files:

```bash
pip install torch transformers python-pptx Pillow
```

To load multiple documents, a path to a directory can also be passed.

```python
from beyondllm.source import fit

data = fit(path="path/to/directory/", dtype="pdf")
```

### **2. UrlLoader:**

&#x20;Extracts text content from web pages given a URL or a list of URLs. Supports chunk\_size and chunk\_overlap parameters. This requires an additional library which can be installed by running:

```bash
pip install llama-index-readers-web
```

* **Multiple Inputs:** Accepts a list of URLs.

**Code Snippet (Loading Multiple Web Pages):**

```python
from beyondllm.source import fit

urls = ["https://www.example.com", "https://www.anotherwebsite.org"]
data = fit(path=urls, dtype="url")
```

### **3. YoutubeLoader:**

Downloads and processes transcripts from YouTube videos. requires the following additional libraries:

```bash
pip install llama-index-readers-web
```

* **Multiple Inputs:** Accepts a list of YouTube video URLs.

**Code Snippet (Loading Transcripts from Multiple Videos):**

```python
from beyondllm.source import fit

youtube_urls = ["https://www.youtube.com/watch?v=video_id_1", "https://www.youtube.com/watch?v=video_id_2"]
data = fit(path=youtube_urls, dtype="youtube")
```

### **4. LlamaParseLoader:**

Leverages the LlamaParse Cloud API to extract structured information and text from various documents. Requires an additional llama\_parse\_key parameter or `LLAMA_CLOUD_API_KEY` environment variable to be set. Supports chunk\_size and chunk\_overlap parameters. We internally perform a markdown splitting on your data.

```bash
pip install llama-parse
```

* **File Types Supported (Currently):** "pdf"
* **Requires:** llama\_parse\_key parameter or LLAMA\_CLOUD\_API\_KEY environment variable (obtain an API key from [https://cloud.llamaindex.ai/login](https://cloud.llamaindex.ai/login)).
* **Multiple Inputs:** Accepts a list of PDF file paths.

**Code Snippet (Loading Multiple PDFs with LlamaParse):**

```python
from beyondllm.source import fit

pdf_paths = ["path/to/document1.pdf", "path/to/document2.pdf"]
data = fit(path=pdf_paths, dtype="llama-parse", llama_parse_key="your_llama_parse_api_key")
```

*   For LlamaParseLoader in Google Colab, run these lines before using fit:

    ```python
    import nest_asyncio
    nest_asyncio.apply()
    ```

**By leveraging the diverse range of loaders in BeyondLLM, you can effectively incorporate information from various sources into your RAG pipeline for enhanced question answering and knowledge retrieval capabilities.**
