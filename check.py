import shutil, importlib.metadata as md
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from models import get_embeddings

print("chromadb", md.version("chromadb"), "| langchain-chroma", md.version("langchain-chroma"))
emb = get_embeddings()
docs = TextLoader("Data/notes.txt").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(docs)

shutil.rmtree("dbg", ignore_errors=True)
db = Chroma.from_documents(chunks, emb, collection_name="notes", persist_directory="dbg")
print("same object :", db._collection.count(), len(db.get()["ids"]))

db2 = Chroma(collection_name="notes", embedding_function=emb, persist_directory="dbg")
print("fresh object:", db2._collection.count(), len(db2.get()["ids"]), db2._collection.name)