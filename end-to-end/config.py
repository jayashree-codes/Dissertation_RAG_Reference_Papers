"""
Central configuration. The retrieval defaults are provisional: run `python evaluate.py`
on your own documents and update them from the results.
"""

from pathlib import Path

# --- Paths ---------------------------------------------------------------
DATA_DIR = Path("data")
PERSIST_DIR = Path("vectorstore")
COLLECTION_NAME = "docs"

# --- Chunking (characters) -----------------------------------------------
# all-MiniLM-L6-v2 reads at most 256 tokens (~1,000 characters of English) and silently
# ignores the rest, so keep chunks comfortably below that.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 160          # 20% of CHUNK_SIZE

# --- Retrieval -----------------------------------------------------------
RETRIEVAL_K = 4
USE_MMR = False              # turn on only if evaluate.py shows it helps on your documents
MMR_LAMBDA = 0.5             # 1.0 = pure relevance, 0.0 = pure diversity

# --- Models ----------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # free, local, 384 dimensions
LLM_PROVIDER = "ollama"      # "ollama" (free, local) or "claude"
OLLAMA_MODEL = "llama3.2"
CLAUDE_MODEL = "claude-haiku-5-5"

# The model is told to reply with exactly this when the context lacks the answer.
REFUSAL = "I don't know based on the provided documents."
