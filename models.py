from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(model = EMBEDDING_MODEL, encode_kwargs = {"normalize_embeddings": True})

def llm(provider = "ollama"):
    if provider == "claude":
        from dotenv import load_dotenv
        from langchain_anthropic import ChatAnthropic
        load_dotenv()
        return ChatAnthropic(model="claude-haiku-5-5", temperature=0)
    return ChatOllama(model="llama3.2", temperature=0)
