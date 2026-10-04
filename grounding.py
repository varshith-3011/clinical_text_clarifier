import re


def normalize_for_grounding(text: str) -> str:
    if text is None:
        return ""

    normalized = re.sub(r"\s+", " ", str(text).strip().lower())
    return normalized


def is_grounded(evidence_text: str, original_text: str) -> bool:
    evidence = normalize_for_grounding(evidence_text)
    original = normalize_for_grounding(original_text)

    if not evidence or not original:
        return False

    return re.search(rf"(?<!\w){re.escape(evidence)}(?!\w)", original) is not None
