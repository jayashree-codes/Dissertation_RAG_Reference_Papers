from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from cleaning import clean

pages = PyPDFLoader("data/Related_papers/Paper1.pdf").load()
for p in pages:
    p.page_content = clean(p.page_content)

print("pages:", len(pages), "| total chars:", sum(len(p.page_content) for p in pages))

for size, overlap in [(500, 50), (1000, 150), (2000, 200)]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = splitter.split_documents(pages)
    avg = sum(len(c.page_content) for c in chunks) / len(chunks)
    print(f"size={size}, overlap={overlap} -> {len(chunks)} chunks, avg {avg:.0f} chars")


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(pages)
for c in chunks[3:5]:
    print("\n---", c.metadata.get("page"), "|", len(c.page_content), "chars")
    print(c.page_content)