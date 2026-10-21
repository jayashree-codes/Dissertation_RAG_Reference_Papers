from langchain_community.document_loaders import TextLoader, PyPDFLoader
import re

txt_docs = TextLoader("data/notes.txt").load()
print("TXT -> ", len(txt_docs),"documents")
print(txt_docs[0].metadata)
print(txt_docs[0].page_content[:300])

pdf_docs = PyPDFLoader("data/Related_papers/Paper1.pdf").load()
print("TXT -> ", len(pdf_docs),"documents")
print(pdf_docs[0].metadata)
print("Before", pdf_docs[0].page_content[:300])

def clean(text: str) -> str:
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
    text = re.sub(r"-\n(\w)", r"\1", text)        # rejoin hyphenated line breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # single newlines -> spaces
    text = re.sub(r"[ \t]+", " ", text)           # collapse repeated spaces
    return text.strip()

for d in pdf_docs:
    d.page_content = clean(d.page_content)

print("After", pdf_docs[1].page_content[:500])
