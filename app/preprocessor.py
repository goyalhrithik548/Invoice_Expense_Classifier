import re


def preprocess(text: str) -> str:
    """
    Clean and normalize invoice text before vectorization.
    Steps: lowercase, remove special chars, collapse whitespace.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
