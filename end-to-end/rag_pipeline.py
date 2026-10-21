"""
Query side of the pipeline: retriever -> numbered context -> prompt -> LLM -> answer.

The chain retrieves ONCE and carries the retrieved chunks through, so the sources shown
to the user are exactly the chunks the model saw.
"""

import re

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import config
import models

SYSTEM = (
    "Answer the question using ONLY the context below. Cite the chunks you used like [1].\n"
    "If the context does not contain the answer, reply exactly: " + config.REFUSAL + "\n\n"
    "Context:\n{context}"
)
PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", "{question}")])


def where(meta) -> str:
    """Human-readable location of a chunk, e.g. 'paper.pdf, p.8-9'."""
    if meta.get("kind") != "pdf":
        return str(meta.get("source", "unknown"))
    page, end = meta.get("page"), meta.get("page_end", meta.get("page"))
    return f"{meta.get('source', 'unknown')}, p.{page}" + (f"-{end}" if end != page else "")


def format_docs(docs) -> str:
    return "\n\n".join(f"[{i}] ({where(d.metadata)})\n{d.page_content}" for i, d in enumerate(docs, 1))


def is_refusal(answer: str) -> bool:
    def norm(s):
        return s.strip().rstrip(".").replace("’", "'").lower()
    return norm(answer) == norm(config.REFUSAL)


def citations(answer: str, docs) -> list:
    """Map [n] markers in the answer back to the chunk (and page) they refer to."""
    numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    return [
        {"n": n, "where": where(docs[n - 1].metadata), "excerpt": docs[n - 1].page_content[:200].strip()}
        for n in numbers if 1 <= n <= len(docs)
    ]


def open_vectorstore():
    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=models.get_embeddings(),
        persist_directory=str(config.PERSIST_DIR),
    )
    check_not_empty(store)
    return store


def check_not_empty(store):
    """Fail loudly. An empty or wrongly-located store otherwise makes every question
    come back as 'I don't know' with no hint of why."""
    if not store.get(limit=1)["ids"]:
        raise RuntimeError(
            f"The vector store is empty (looked in {config.PERSIST_DIR}/, collection "
            f"'{config.COLLECTION_NAME}'). Run `python ingest.py` first, from the project folder."
        )


def make_retriever(store, k=config.RETRIEVAL_K, mmr=config.USE_MMR, mmr_lambda=config.MMR_LAMBDA):
    if mmr:
        return store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": max(20, 4 * k), "lambda_mult": mmr_lambda},
        )
    return store.as_retriever(search_type="similarity", search_kwargs={"k": k})


class RAGPipeline:
    def __init__(self, vectorstore=None, llm=None, k=config.RETRIEVAL_K,
                 mmr=config.USE_MMR, mmr_lambda=config.MMR_LAMBDA):
        self.store = vectorstore if vectorstore is not None else open_vectorstore()
        check_not_empty(self.store)
        self.retriever = make_retriever(self.store, k, mmr, mmr_lambda)
        self._llm = llm
        self._chain = None

    def retrieve(self, question):
        return self.retriever.invoke(question)

    def _get_chain(self):
        if self._chain is None:
            llm = self._llm or models.get_llm()
            self._chain = (
                RunnablePassthrough.assign(docs=lambda x: self.retrieve(x["question"]))
                .assign(context=lambda x: format_docs(x["docs"]))
                .assign(answer=PROMPT | llm | StrOutputParser())
            )
        return self._chain

    def ask(self, question: str) -> dict:
        out = self._get_chain().invoke({"question": question})
        answer = out["answer"].strip()
        return {
            "question": question,
            "answer": answer,
            "docs": out["docs"],
            "refused": is_refusal(answer),
            "citations": citations(answer, out["docs"]),
        }
