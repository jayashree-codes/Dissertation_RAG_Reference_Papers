"""
Evaluation harness. Measures the pipeline in layers so a failure can be placed:

  evidence intact : after chunking, does every evidence phrase still exist whole inside
                    some chunk? (tests the CHUNKER)
  context hit     : do the top-k retrieved chunks contain ALL the evidence phrases?
                    (tests the EMBEDDINGS + RETRIEVER)
  page hit        : does any retrieved chunk overlap a gold page? (looser, for citations)
  answer / refusal: does the LLM's answer contain the expected facts, and does it refuse
                    exactly when the paper cannot answer? (tests the PROMPT + LLM)

Modes
  python evaluate.py                      retrieval grid over chunk size / k / MMR (free, no LLM)
  python evaluate.py --generate --size 800 --k 4 --mmr off --provider ollama
                                          end-to-end run for one configuration

Put ONLY the documents the eval set was written for in --data-dir (e.g. just Paper1.pdf).
"""

import argparse
import json
import re
import uuid
from pathlib import Path

from langchain_chroma import Chroma

import config
import models
from ingest import build_chunks
from rag_pipeline import RAGPipeline, make_retriever


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def load_questions(path):
    return json.load(open(path, encoding="utf-8"))["questions"]


def build_store(chunks, emb):
    """In-memory store with a unique collection name, so grid runs never collide."""
    return Chroma.from_documents(
        chunks, emb,
        collection_name=f"eval_{uuid.uuid4().hex[:8]}",
        ids=[c.metadata["chunk_id"] for c in chunks],
    )


def evidence_in(question, text: str) -> bool:
    return all(norm(e["text"]) in text for e in question["evidence"])


def page_overlap(docs, gold_pages) -> bool:
    gold = set(gold_pages)
    return any(set(range(d.metadata["page"], d.metadata["page_end"] + 1)) & gold for d in docs)


# --------------------------------------------------------------------------- retrieval grid
def retrieval_grid(questions, data_dir, sizes, ks):
    emb = models.get_embeddings()
    answerable = [q for q in questions if q["type"] == "answerable"]
    n = len(answerable)
    rows = []
    for size in sizes:
        chunks = build_chunks(data_dir, size)
        texts = [norm(c.page_content) for c in chunks]
        intact = sum(all(any(norm(e["text"]) in t for t in texts) for e in q["evidence"]) for q in answerable)
        store = build_store(chunks, emb)
        for k in ks:
            for use_mmr in (False, True):
                retriever = make_retriever(store, k=k, mmr=use_mmr)
                ctx_hits = page_hits = chars = 0
                missed = []
                for q in answerable:
                    docs = retriever.invoke(q["question"])
                    ctx = norm(" ".join(d.page_content for d in docs))
                    chars += sum(len(d.page_content) for d in docs)
                    if evidence_in(q, ctx):
                        ctx_hits += 1
                    else:
                        missed.append(q["id"])
                    page_hits += page_overlap(docs, q["gold_pages"])
                rows.append(dict(size=size, k=k, mmr="on" if use_mmr else "off", chunks=len(chunks),
                                 intact=f"{intact}/{n}", ctx=f"{ctx_hits}/{n}", page=f"{page_hits}/{n}",
                                 avg_chars=round(chars / n), missed=", ".join(missed) or "-"))
    return rows


def print_grid(rows):
    head = "| chunk size | k | MMR | chunks | evidence intact | context hit | page hit | avg context chars | missed |"
    lines = [head, "|" + "---|" * 9]
    for r in rows:
        lines.append(f"| {r['size']} | {r['k']} | {r['mmr']} | {r['chunks']} | {r['intact']} | {r['ctx']} | "
                     f"{r['page']} | {r['avg_chars']} | {r['missed']} |")
    table = "\n".join(lines)
    print(table)
    return table


# --------------------------------------------------------------------------- generation
def answer_ok(question, answer: str) -> bool:
    a = answer.lower()
    return all(any(alt.lower() in a for alt in group) for group in question["must_contain"])


def generation_eval(questions, data_dir, size, k, use_mmr, provider):
    emb = models.get_embeddings()
    store = build_store(build_chunks(data_dir, size), emb)
    pipe = RAGPipeline(vectorstore=store, llm=models.get_llm(provider), k=k, mmr=use_mmr)

    results, counts = [], dict(pass_=0, retrieval_miss=0, generation_fail=0, refused_ok=0, refused_wrong=0)
    for q in questions:
        res = pipe.ask(q["question"])
        if q["type"] == "answerable":
            ctx = norm(" ".join(d.page_content for d in res["docs"]))
            has_ctx = evidence_in(q, ctx)
            if not res["refused"] and answer_ok(q, res["answer"]):
                verdict = "pass"; counts["pass_"] += 1
            elif not has_ctx:
                verdict = "retrieval miss"; counts["retrieval_miss"] += 1
            else:
                verdict = "generation fail"; counts["generation_fail"] += 1
        else:
            verdict = "refused (correct)" if res["refused"] else "ANSWERED (should refuse)"
            counts["refused_ok" if res["refused"] else "refused_wrong"] += 1
        results.append(dict(id=q["id"], type=q["type"], verdict=verdict, question=q["question"],
                            answer=res["answer"], retrieved=[d.metadata["chunk_id"] for d in res["docs"]]))
        print(f"{q['id']}  {verdict:<26} {res['answer'][:90]!r}")

    n_ans = sum(q["type"] == "answerable" for q in questions)
    n_un = len(questions) - n_ans
    print(f"\nAnswerable : {counts['pass_']}/{n_ans} correct | {counts['retrieval_miss']} retrieval misses | "
          f"{counts['generation_fail']} generation failures (incl. false refusals)")
    print(f"Unanswerable: {counts['refused_ok']}/{n_un} correctly refused")
    return results, counts


# --------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-set", default="eval_set.json")
    ap.add_argument("--data-dir", default=str(config.DATA_DIR))
    ap.add_argument("--sizes", type=int, nargs="+", default=[500, 800, 1000])
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--size", type=int, default=config.CHUNK_SIZE)
    ap.add_argument("--k", type=int, default=config.RETRIEVAL_K)
    ap.add_argument("--mmr", choices=["on", "off"], default="on" if config.USE_MMR else "off")
    ap.add_argument("--provider", default=config.LLM_PROVIDER, choices=["ollama", "claude"])
    args = ap.parse_args(argv)

    questions = load_questions(args.eval_set)
    if args.generate:
        results, counts = generation_eval(questions, args.data_dir, args.size, args.k,
                                          args.mmr == "on", args.provider)
        Path("results_generation.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print("\nFull answers saved to results_generation.json")
    else:
        rows = retrieval_grid(questions, args.data_dir, args.sizes, args.ks)
        table = print_grid(rows)
        Path("results_retrieval.md").write_text(table + "\n")
        print("\nTable saved to results_retrieval.md")


if __name__ == "__main__":
    main()
