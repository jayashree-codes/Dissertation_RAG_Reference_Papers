"""
The only file that names a model vendor. Imports are lazy so that, for example, running
the tests does not require torch or an API key.
"""

import config


def get_embeddings():
    """Local sentence-transformers model. Vectors are unit length, so dot product = cosine."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_llm(provider=None):
    provider = provider or config.LLM_PROVIDER
    if provider == "ollama":
        from langchain_ollama import ChatOllama      # needs the Ollama app/server running

        return ChatOllama(model=config.OLLAMA_MODEL, temperature=0)
    if provider == "claude":
        from dotenv import load_dotenv
        from langchain_anthropic import ChatAnthropic  # needs ANTHROPIC_API_KEY in .env

        load_dotenv()
        return ChatAnthropic(model=config.CLAUDE_MODEL, temperature=0)
    raise ValueError(f"Unknown provider {provider!r}; use 'ollama' or 'claude'.")
