from .loaders.simpleFileLoader import SimpleLoader
# from .loaders.llamaParseLoader import SmartLlamaParseLoader

def fit(path, loader_type='simple', config_file=None, **kwargs):
    
    if loader_type == 'simple' or path.endswith('.pdf'):
        loader = SimpleLoader(config_file=config_file, **kwargs)
    # elif loader_type == 'llama-parse':
    #     loader = SmartLlamaParseLoader(config_file=config_file, **kwargs)
    else:
        raise NotImplementedError(f"Loader for the type '{loader_type}' is not implemented.")
    
    return loader.fit(path)
