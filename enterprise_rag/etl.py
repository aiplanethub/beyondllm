def fit(path,loader_type="pdf",**kwargs):
    """
    Fits the data from the given path using the appropriate loader based on the specified loader type.
    
    Parameters:
        path (str): The path to the data to be loaded.
        loader_type (str): Specifies the type of loader to use based on the file type or source.
        **kwargs: Additional keyword arguments to be passed to the selected loader.
    
    Returns:
        The loaded and processed data by the specified loader.

    Raises:
        NotImplementedError: If the loader_type is not supported.
    """

    simple_loader_file_types = [
        "pdf", "csv", "docx", "epub", "hwp", "ipynb",
        "jpg", "jpeg", "mbox", "md", "mp3", "mp4", 
        "png", "ppt", "pptx", "pptm"
    ]

    if loader_type in simple_loader_file_types and loader_type!='llama-parse':
        from .loaders.simpleLoader import SimpleLoader
        loader = SimpleLoader(path,**kwargs)
    elif loader_type=="url":
        from .loaders.urlLoader import UrlLoader
        loader = UrlLoader(path,**kwargs)
    elif loader_type=="youtube":
        from .loaders.youtubeLoader import YoutubeLoader
        loader = YoutubeLoader(path,**kwargs)
    elif loader_type == 'llama-parse':
        from .loaders.llamaParseLoader import LlamaParseLoader
        loader = LlamaParseLoader(path,**kwargs)
    else:
        raise NotImplementedError(f"Loader for the type '{loader_type}' is not implemented.")
    
    return loader.fit(path)
