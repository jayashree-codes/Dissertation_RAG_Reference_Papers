"""
Interactive Q&A over your documents. Run `python ingest.py` first.

    python main.py
    python main.py --provider claude --k 5 --mmr on
"""

import argparse

import config
import models
from rag_pipeline import RAGPipeline, where


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default=config.LLM_PROVIDER, choices=["ollama", "claude"])
    ap.add_argument("--k", type=int, default=config.RETRIEVAL_K)
    ap.add_argument("--mmr", choices=["on", "off"], default="on" if config.USE_MMR else "off")
    args = ap.parse_args(argv)

    pipeline = RAGPipeline(llm=models.get_llm(args.provider), k=args.k, mmr=args.mmr == "on")
    print("Ready. Ask a question about your documents (Ctrl+C to quit).\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not question:
            continue

        result = pipeline.ask(question)
        print(f"\nAssistant: {result['answer']}\n")
        if result["citations"]:
            print("Cited:")
            for c in result["citations"]:
                print(f"  [{c['n']}] {c['where']}: \"{c['excerpt'][:120]}...\"")
        print("Retrieved:", "; ".join(f"[{i}] {where(d.metadata)}" for i, d in enumerate(result["docs"], 1)))
        print()


if __name__ == "__main__":
    main()
