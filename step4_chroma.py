import shutil 
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from models import get_embeddings

emb = get_embeddings()
docs = TextLoader("Data/notes.txt").load()
chunks = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap=50, add_start_index=True).split_documents(docs)
print("chunks:", len(chunks))

def count(path):
    db = Chroma(collection_name = "notes", embedding_function = emb, persist_directory = path)
    return len(db.get()["ids"])

shutil.rmtree("step4_db", ignore_errors=True)
db = Chroma.from_documents(chunks, emb, collection_name = "notes", persist_directory = "step4_db")
print("stored:", count("step4_db"))
rec = db.get(limit=1, include=["metadatas", "embeddings", "documents"])
print("id: ", rec["ids"][0])
print("text:", rec["documents"][0][:60], "...")
print("metadata:", rec["metadatas"][0])
print("vector length:", len(rec["embeddings"][0]))

for q in ["When does Aurora launch?",
          "What is the deadline for going live?",
          "What is the weather like in Edinburgh?"]:
    print("\nQ:", q)
    for doc, dist in db.similarity_search_with_score(q, k=3):
        print(f"  dist={dist:.3f}  cos={1 - dist/2:.3f}  {doc.page_content[:30]!r}")

db2 = Chroma(collection_name="notes", embedding_function=emb, persist_directory="step4_db")
print("\nreloaded:", db2.similarity_search("Who knows the billing system?", k=1)[0].page_content[:60])

shutil.rmtree("step4_dup", ignore_errors=True)
for _ in range(2):
    Chroma.from_documents(chunks, emb, collection_name="notes", persist_directory="step4_dup")
print("\nrandom ids, ingested twice:", count("step4_dup"))

shutil.rmtree("step4_ids", ignore_errors=True)
ids = [f"{c.metadata['source']}:{c.metadata['start_index']}" for c in chunks]
for _ in range(2):
    Chroma.from_documents(chunks, emb, ids=ids, collection_name="notes", persist_directory="step4_ids")
print("stable ids, ingested twice:", count("step4_ids"))

