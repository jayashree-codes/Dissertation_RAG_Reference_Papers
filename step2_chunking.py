from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = TextLoader("data/notes.txt").load()
print("total chars:", len(docs[0].page_content))

for size, overlap in [(300,0),(300,50),(150,30)]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = size, chunk_overlap = overlap, add_start_index = True)
    chunks = text_splitter.split_documents(docs)
    print(f"\n=== size={size}, overlap={overlap} -> {len(chunks)} chunks ===")
    for i, c in enumerate(chunks):
        print(f"--- chunk {i} | {len(c.page_content)} chars | metadata={c.metadata}")
        print(c.page_content)

