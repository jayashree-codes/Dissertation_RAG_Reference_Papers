# chunk_pdf.py
from bisect import bisect_right
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from cleaning import clean

PDF = "Data/Related_papers/Paper1.pdf"
pages = PyPDFLoader(PDF).load()

# 1. Join cleaned pages, remembering where each page starts
text, page_starts = "", []
for p in pages:
    page_starts.append(len(text))
    text += clean(p.page_content) + " "

# 2. Drop the reference list (heuristic: last "References" in the final 40%)
cut = text.rfind("References")
if cut > 0.6 * len(text):
    text = text[:cut]

# 3. Sentence-aware chunking over the whole document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=300,
    separators=["\n\n", r"(?<=[.!?])\s+", " ", ""],
    is_separator_regex=True, add_start_index=True,
)
chunks = splitter.create_documents([text], metadatas=[{"source": PDF}])

# 4. Map each chunk back to the page(s) it came from
for c in chunks:
    start = c.metadata["start_index"]
    end = start + len(c.page_content) - 1
    c.metadata["page"] = pages[bisect_right(page_starts, start) - 1].metadata["page"]
    c.metadata["page_end"] = pages[bisect_right(page_starts, end) - 1].metadata["page"]

print(len(chunks), "chunks")
for c in chunks[3:5]:
    print("\n---", c.metadata["page"], "to", c.metadata["page_end"], "|", len(c.page_content), "chars")
    print(c.page_content)