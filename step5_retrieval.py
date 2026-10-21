import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from models import get_embeddings

emb = get_embeddings()
docs = TextLoader("Data/notes.txt").load()

def split(chunk_size, chunk_overlap):
    return RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap, add_start_index = True).split_documents(docs)

small = split(150,30)
shutil.rmtree("s5_small", ignore_errors=True)
db = Chroma.from_documents(small, emb, collection_name = "notes", persist_directory = "s5_small", ids=[f"{c.metadata['source']}:{c.metadata['start_index']}"for c in small])
for i, c in enumerate(small):
    print(i, repr(c.page_content[:70]))

for q, answer in [("When does Aurora launch?", "15 June"),
                  ("What is the Aurora budget?", "40,000")]:
    print("\nQ:", q)
    for rank, (d, dist) in enumerate(db.similarity_search_with_score(q, k=4), 1):
        print(f"  rank {rank}  cos={1 - dist/2:.3f}  has_answer={answer in d.page_content}  {d.page_content[:80]!r}")
print("PART B ===========================================================")
big = split(300, 50)
shutil.rmtree("s5_dup", ignore_errors=True)
for _ in range(2):
    dup = Chroma.from_documents(big, emb, collection_name="notes", persist_directory="s5_dup")

q = "When does Aurora launch?"
sim = dup.as_retriever(search_type="similarity", search_kwargs={"k": 3})
mmr = dup.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 8, "lambda_mult": 0.5})
print("\nsimilarity:", [d.page_content[:15] for d in sim.invoke(q)])
print("mmr       :", [d.page_content[:15] for d in mmr.invoke(q)])
print("PART C ===========================================================")
for lam in (1.0, 0.5, 0.0):
    r = db.as_retriever(search_type="mmr",
                        search_kwargs={"k": 3, "fetch_k": len(small), "lambda_mult": lam})
    print(f"\nlambda={lam}:", [d.page_content[:20] for d in r.invoke(q)])

