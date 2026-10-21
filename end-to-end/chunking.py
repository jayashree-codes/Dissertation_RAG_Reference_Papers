"""
Cleaning and chunking. Turns raw extracted pages into chunks that know which page(s)
they came from.

Design notes (each one fixes something observed while building this):
- pypdf output contains no blank lines at all, so section headings are the only reliable
  paragraph boundaries. We insert a blank line before each heading so the splitter
  prefers to cut there.
- A single newline in a PDF means "line wrapped", not "paragraph ended", so it becomes a space.
- The whole document is chunked as one string, then each chunk is mapped back to its page(s)
  with a binary search. Chunking page by page would cut sentences at every page boundary.
- Page numbers are 1-indexed physical pages of the PDF.
"""

import re
from bisect import bisect_right

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

ARXIV_STAMP = re.compile(r"arXiv:\d{4}\.\d{4,5}v\d+\s+\[[^\]]+\]\s+\d{1,2}\s+\w{3}\s+\d{4}")
LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}

# A line that looks like a section heading: "2.3 UNETR", "1 Introduction", or a bare known heading.
HEADING = re.compile(
    r"(?m)^(?=(?:\d+(?:\.\d+){0,2}[ \t]+[A-Za-z][^\n.]{1,44}[^\n.\-]"
    r"|Abstract|Keywords|Acknowledgements|References)[ \t]*$)"
)
REFERENCES = re.compile(r"(?m)^[ \t]*References[ \t]*$")


def clean(text: str) -> str:
    """Repair common PDF-extraction damage. May return text starting with newlines (a heading)."""
    text = ARXIV_STAMP.sub("", text)
    for bad, good in LIGATURES.items():
        text = text.replace(bad, good)
    text = re.sub(r"(?m)^[ \t]*\d{1,3}[ \t]*(?:\n|\Z)", "", text)   # lines that are ONLY a page number
    text = re.sub(r"-\n(?=[a-z])", "", text)                  # "optimi-\nsation" -> "optimisation"
    text = re.sub(r"-\n(?=[A-Z])", "-", text)                 # "U-\nNet" -> "U-Net"
    text = HEADING.sub("\n\n", text)                          # blank line before each heading
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)              # a lone newline is just a line wrap
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<=\d) \.(?=\d)", ".", text)             # kerning splits "0.78802" into "0 .78802"
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.lstrip(" \t").rstrip()


def strip_references(pages):
    """Drop the bibliography: everything after the last 'References' heading, if it sits
    in the final 40% of the document. Truncating the end keeps all earlier offsets valid."""
    total = sum(len(t) for _, t in pages)
    cut, offset = None, 0
    for i, (_, t) in enumerate(pages):
        for m in REFERENCES.finditer(t):
            if offset + m.start() > 0.6 * total:
                cut = (i, m.start())
        offset += len(t)
    if cut is None:
        return pages
    i, pos = cut
    return pages[:i] + [(pages[i][0], pages[i][1][:pos])]


def join_pages(pages):
    """Clean each page and join into one string. Returns (text, page_starts, page_numbers),
    where page_starts[i] is the offset in `text` of the first character of page_numbers[i]."""
    text, starts, numbers = "", [], []
    for number, raw in pages:
        cleaned = clean(raw)
        body = cleaned.lstrip("\n")
        if not body:
            continue
        if text:
            text = text.rstrip(" ") + ("" if cleaned.startswith("\n") else " ")
        starts.append(len(text) + (len(cleaned) - len(body)))
        numbers.append(number)
        text += cleaned
    return text, starts, numbers


def make_chunks(pages, source, size, overlap, kind="pdf"):
    """pages: list of (page_number, raw_text). Returns Documents with metadata:
    source, kind, page, page_end, start_index, chunk_id (stable across runs)."""
    if kind == "pdf":
        pages = strip_references(pages)
    text, starts, numbers = join_pages(pages)
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", r"(?<=[.!?])\s+", " ", ""],
        is_separator_regex=True,
        add_start_index=True,
    )
    chunks = splitter.create_documents([text], metadatas=[{"source": source, "kind": kind}])
    for c in chunks:
        start = c.metadata["start_index"]
        end = start + len(c.page_content) - 1
        c.metadata["page"] = numbers[bisect_right(starts, start) - 1]
        c.metadata["page_end"] = numbers[bisect_right(starts, end) - 1]
        c.metadata["chunk_id"] = f"{source}:{start}"
    return chunks
