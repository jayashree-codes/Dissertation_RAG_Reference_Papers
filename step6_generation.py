import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from models import get_embeddings, llm

PROVIDER = "ollama"
chunk_size, chunk_overlap, K = 150, 30, 1

emb = get_embeddings()
docs = TextLoader("Data/notes.txt").load()
chunks = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap, add_start_index = True).split_documents(docs)

shutil.rmtree("s6_db",ignore_errors = True)
db = Chroma.from_documents(chunks, emb, collection_name = "notes", persist_directory = "s6_db", ids = [f"{c.metadata['source']}:{c.metadata['start_index']}" for c in chunks])
retriever = db.as_retriever(search_type = "mmr", search_kwargs={"k": K, "fetch_k":len(chunks), "lambda_mult":0.5})

SYSTEM = """Answer the question using ONLY the context below. Cite the chunks you used like [1].
If the context does not contain the answer, reply exactly: I don't know based on the provided documents.

Context:
{context}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM),("human", "{question}")])

def format_docs(docs):
    return "\n\n".join(
        f"[{i}] (source: {d.metadata.get('source')})\n{d.page_content}"
        for i, d in enumerate(docs, 1))

q = "When does Aurora launch?"
text = PROMPT.invoke({"context": format_docs(retriever.invoke(q)),"question": q}).to_string()
print(text)
print("\nrough size:", len(text), "chars ~", len(text) // 4, "tokens")

llm = llm(PROVIDER)
rag = (
    RunnablePassthrough.assign(docs=lambda x:retriever.invoke(x["question"]))
    .assign(context=lambda x: format_docs(x["docs"]))
    .assign(answer=PROMPT | llm | StrOutputParser())
)
for q in ["When does Aurora launch?",
          "Who could block the integration work, and why?",
          "What is the weather like in Edinburgh?",
          "How many engineers are on the Aurora team?"]:
    out = rag.invoke({"question": q})
    print("\nQ:", q)
    print("A:", out["answer"])
    print("sources:", [d.page_content[:25] for d in out["docs"]])