import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from models import get_embeddings, EMBEDDING_MODEL

emb = get_embeddings()

v = np.array(emb.embed_query("When does Aurora launch?"))
print("dimensions:", v.shape[0])
print("first 5 numbers:", v[:5].round(4))
print("length (L2 norm):", np.linalg.norm(v).round(4))

data = TextLoader("Data/notes.txt").load()
chunks = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 50).split_documents(data)
chunk_vecs = np.array(emb.embed_documents(c.page_content for c in chunks))

questions = [
    "When does Aurora launch?",
    "What is the deadline for going live?",
    "Who could block the integration work?",
    "What is the weather like in Edinburgh?",
]

for q in questions:
    query_vecs = np.array(emb.embed_query(q))
    scores = chunk_vecs @ query_vecs
    print(f"\nQ: {q}")
    for i in np.argsort(-scores):
        print(f"  {scores[i]:.3f}  chunk {i}: {chunks[i].page_content[:70]!r}")

print("\nmax tokens per input:", SentenceTransformer(EMBEDDING_MODEL).max_seq_length)
filler = "The weather report covered rainfall and wind. " * 60
fact = "The launch date is 15 June."
q = np.array(emb.embed_query("When is the launch date?"))
texts = {"fact first": fact + " " + filler,
         "fact last": filler + " " + fact,
         "filler only": filler}
for name, t in texts.items():
    print(f"{name:12s} {np.array(emb.embed_documents([t]))[0] @ q:.3f}")
