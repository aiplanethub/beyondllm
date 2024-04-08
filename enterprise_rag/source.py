from .loaders.simpleLoader import SimpleLoader

def fit(path,dtype,**kwargs):
    """
    Fits the data from the given path using the appropriate loader based on the specified loader type.
    
    Parameters:
        path (str): The path to the data to be loaded.
        dtype (str): Specifies the type of loader to use based on the file type or source.
        **kwargs: Additional keyword arguments to be passed to the selected loader.
    
    Returns:
        The loaded and processed data List[TextNode] by the specified loader.

    Raises:
        NotImplementedError: If the dtype is not supported.

    Example:
    from enterprise_rag.source import fit
    data = fit("<youtube-video-url>",dtype="youtube",chunk_size=512,chunk_overlap=100)
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
    else:
        raise NotImplementedError(f"Loader for the type '{dtype}' is not implemented.")
    
    return loader.fit(path)