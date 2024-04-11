from .loaders.simpleLoader import SimpleLoader

def fit(path,dtype,**kwargs):
    """
    Fits the data from the given path(s) using the appropriate loader based on the specified loader type.
    
    Parameters:
        paths (str or List[str]): The path(s) to the data to be loaded. Can be a single path or a list of paths.
        dtype (str): Specifies the type of loader to use based on the file type or source.
        **kwargs: Additional keyword arguments to be passed to the selected loader.
    
    Returns:
        The loaded and processed data List[TextNode] by the specified loader.

    Raises:
        NotImplementedError: If the dtype is not supported.

    Example:
    from enterprise_rag.source import fit
    # For a single path
    pdf_data = fit("/path/to/your/pdf.pdf", dtype="pdf", chunk_size=1024, chunk_overlap=200)
    
    # For multiple paths
    multiple_data = fit(["/path/to/your/pdf1.pdf", "/path/to/your/pdf2.pdf"], dtype="pdf")
    """

    simple_loader_file_types = [
        "pdf", "csv", "docx", "epub", "hwp",
        "mbox", "md", "ppt", "pptx", "pptm"
    ]
    
    if dtype in simple_loader_file_types and dtype!='llama-parse':
        loader = SimpleLoader(path,**kwargs)
    elif dtype=="url":
        from .loaders.urlLoader import UrlLoader
        loader = UrlLoader(path,**kwargs)
    elif dtype=="youtube":
        from .loaders.youtubeLoader import YoutubeLoader
        loader = YoutubeLoader(path,**kwargs)
    elif dtype == 'llama-parse':
        from .loaders.llamaParseLoader import LlamaParseLoader
        loader = LlamaParseLoader(path,**kwargs)
    elif dtype == 'notion':
        from .loaders.notionLoader import NotionLoader
        loader = NotionLoader(path,**kwargs)
    else:
        raise NotImplementedError(f"Loader for the type '{dtype}' is not implemented.")
    
    return loader.fit(path)
