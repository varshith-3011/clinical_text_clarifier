import re


def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)
    cleaned = re.sub(r"\s+", " ", text)
    return cleaned.strip()
