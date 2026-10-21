# Document Q&A with RAG (LangChain + Chroma, runs locally)

Ask questions about your own PDFs and get answers that **cite the page they came from**
and **refuse to guess** when the documents don't contain the answer.

The embedding model runs locally (free). The LLM is either a local model through Ollama
(free) or Claude through the API. A small evaluation harness measures each layer
separately, so when an answer is wrong you can tell whether the chunker, the retriever
or the LLM was at fault.

## How it works

```
ingest.py   PDF/TXT/MD -> clean -> chunk (with page numbers) -> embed -> Chroma
main.py     question -> retrieve top-k chunks -> numbered context -> prompt -> LLM -> answer + citations
evaluate.py measures chunking, retrieval and generation against a hand-checked question set
```

| File | Role |
|---|---|
| `config.py` | every tunable setting in one place |
| `models.py` | the only file that names a model vendor (embeddings + LLM) |
| `chunking.py` | PDF cleaning, heading-aware chunking, chunk -> page mapping |
| `ingest.py` | builds the vector store (rebuilds from scratch on every run) |
| `rag_pipeline.py` | retriever, prompt, LCEL chain, citation parsing, empty-store guard |
| `main.py` | interactive CLI |
| `evaluate.py` + `eval_set.json` | layered evaluation: 10 answerable + 3 unanswerable questions |
| `tests/` | unit tests (fake embeddings and LLM: no downloads, no keys) |

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

mkdir -p data && cp /path/to/your.pdf data/
python ingest.py                      # first run downloads the ~90 MB embedding model
python main.py                        # ask questions
python -m pytest -q                   # run the tests
```

LLM options (set `LLM_PROVIDER` in `config.py` or pass `--provider`):

- **Ollama (free, local):** install from ollama.com, run `ollama pull llama3.2`, keep the app running.
- **Claude:** put `ANTHROPIC_API_KEY=sk-ant-...` in `.env` (Console API key, not a claude.ai login).

## Design decisions, and what each one fixes

Each of these came from a failure observed while building the project.

- **Chunk the whole document, then map chunks back to pages.** Chunking page by page cut
  sentences at every page boundary. Chunking one joined string and recovering `page` /
  `page_end` with a binary search over page start offsets keeps sentences whole and lets a
  citation say "p.8-9".
- **Heading-aware splitting.** `pypdf` output has no blank lines at all, so section headings
  are the only reliable paragraph boundaries; a blank line is inserted before each heading
  and the splitter prefers to cut there.
- **PDF cleaning.** Strips the arXiv margin stamp, fixes ligatures, rejoins hyphenated
  line breaks, drops lone page numbers, and repairs decimals that extraction splits
  (`0 .78802` -> `0.78802`; about 55 of them in the test paper, which made exact-number
  questions unanswerable). Cuts the bibliography so reference entries cannot be retrieved.
- **Chunk size below the embedding window.** `all-MiniLM-L6-v2` reads at most 256 tokens
  (~1,000 characters) and silently ignores the rest.
- **Rebuild on every ingest.** With random IDs, ingesting twice doubled the store (4 -> 8
  records); with stable IDs alone, an edited document would leave stale chunks behind.
- **Fail loudly on an empty store.** A wrong path otherwise produces an empty database and
  every question answers "I don't know" with no hint why.
- **Retrieve once.** The chain carries the retrieved chunks through, so the sources shown
  are exactly what the LLM read.
- **Numbered chunks and an exact refusal sentence.** Numbering lets the model cite `[n]`,
  which maps back to a page; the fixed refusal string makes refusals machine-checkable.

## Evaluation

Three layers, because they fail differently:

| Layer | Metric | What it tells you |
|---|---|---|
| Chunker | *evidence intact*: every evidence phrase survives whole inside some chunk | whether chunking damaged the answer |
| Retriever | *context hit*: the top-k chunks together contain all evidence phrases; *page hit*: a retrieved chunk overlaps a gold page | whether the right text reached the prompt |
| LLM | answer contains the expected facts; unanswerable questions get the exact refusal | whether the model used the context correctly |

```bash
mkdir eval_data && cp data/Paper1.pdf eval_data/          # ONLY the document the eval set is about
python evaluate.py --data-dir eval_data                    # retrieval grid, free, no LLM
python evaluate.py --data-dir eval_data --generate --size 800 --k 4 --mmr off --provider ollama
```

In generation mode every answerable question gets a verdict: `pass`, `retrieval miss`
(the evidence never reached the prompt), or `generation fail` (the evidence was there but
the answer was wrong or a false refusal).

### Results

*Not yet filled in. Run the commands above on your machine and paste the table here, then
set the defaults in `config.py` from it.*

| chunk size | k | MMR | chunks | evidence intact | context hit | page hit | avg context chars | missed |
|---|---|---|---|---|---|---|---|---|
| 500 | 3 | off | 101 | 10/10 | 4/10 | 8/10 | 1162 | q01, q03, q06, q07, q08, q09 |
| 500 | 3 | on | 101 | 10/10 | 4/10 | 9/10 | 1143 | q01, q03, q06, q07, q08, q09 |
| 500 | 5 | off | 101 | 10/10 | 5/10 | 9/10 | 1933 | q01, q03, q06, q07, q09 |
| 500 | 5 | on | 101 | 10/10 | 5/10 | 9/10 | 1902 | q01, q03, q06, q08, q09 |
| 800 | 3 | off | 64 | 10/10 | 6/10 | 7/10 | 1941 | q01, q06, q07, q09 |
| 800 | 3 | on | 64 | 10/10 | 6/10 | 8/10 | 1840 | q01, q06, q08, q09 |
| 800 | 5 | off | 64 | 10/10 | 7/10 | 9/10 | 3266 | q01, q07, q09 |
| 800 | 5 | on | 64 | 10/10 | 6/10 | 8/10 | 2861 | q01, q06, q08, q09 |
| 1000 | 3 | off | 52 | 10/10 | 4/10 | 7/10 | 2511 | q01, q02, q03, q06, q07, q09 |
| 1000 | 3 | on | 52 | 10/10 | 5/10 | 8/10 | 2261 | q01, q02, q06, q08, q09 |
| 1000 | 5 | off | 52 | 10/10 | 5/10 | 8/10 | 4103 | q01, q02, q03, q06, q09 |
| 1000 | 5 | on | 52 | 10/10 | 5/10 | 8/10 | 3815 | q01, q02, q06, q08, q09 |

### Limits of this evaluation

- Ten answerable questions is a small sample; a difference of one question is not evidence.
- Answer checking is substring matching, so a correct answer phrased unusually can fail and
  a vague one can pass. Read the failures by hand (`results_generation.json`).
- The evaluation set was written for one paper; other document types need their own.
- Tables extracted from PDFs lose their column structure, so table questions are the
  hardest and are included on purpose.

## Extending it

- Add hybrid search (BM25 + embeddings) if exact identifiers or rare names are retrieved poorly.
- Add a cross-encoder reranker: retrieve ~20 candidates, re-order, keep the best 4.
- Prepend the document title to each chunk before embedding (contextual retrieval).
- For scanned PDFs, add OCR; `ingest.py` already warns about pages with no extractable text.
