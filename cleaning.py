# cleaning.py
import re

ARXIV_STAMP = re.compile(r"arXiv:\d{4}\.\d{4,5}v\d+\s+\[[^\]]+\]\s+\d{1,2}\s+\w{3}\s+\d{4}")


def clean(text: str) -> str:
    text = ARXIV_STAMP.sub("", text)
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()