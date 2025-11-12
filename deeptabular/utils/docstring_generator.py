from pretab.preprocessor import Preprocessor


def generate_docstring(config, model_description, examples):
    """Generates the complete docstring for any model class by combining config and Preprocessor docstrings.

    The `Parameters` tag is stripped from the Preprocessor docstring to avoid duplication.
    """
    config_doc = config.__doc__ or "No documentation for DefaultFTTransformerConfig."
    preprocessor_doc = Preprocessor.__doc__ or "No documentation for Preprocessor."

    # Remove "Parameters" section header from the Preprocessor docstring
    preprocessor_doc_cleaned = preprocessor_doc.split("Parameters\n    ----------\n", 1)[-1].strip()

    preprocessor_doc_cleaned = preprocessor_doc_cleaned.split("Attributes")[0].strip()

    config_doc += preprocessor_doc_cleaned

    return f"""
    {model_description.strip()}

    Notes
    -----
    The parameters for this class include the attributes from the config
    dataclass as well as preprocessing arguments handled by the base class.

    {config_doc}

    Examples
    --------
    {examples.strip()}
    """
