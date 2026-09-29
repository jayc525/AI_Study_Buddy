import re
from typing import List


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}


COMMON_OCR_FIXES = {
    "sgnal": "signal",
    "sgnals": "signals",
    "signaly": "signals",
    "syptem": "system",
    "systen": "system",
    "alysis": "analysis",
    "definced": "defined",
    "mathemateally": "mathematically",
    "nathemateally": "mathematically",
    "enginering": "engineering",
    "ramesignal": "ramp signal",
    "dmpulse": "impulse",
    "observa": "observable",
    "bie": "be",
    "chang": "change",
    "caninuow": "continuous",
    "continuos": "continuous",
    "disore": "discrete",
    "prsorete": "discrete",
    "axt": "at",
    "tine": "time",
    "dndapenaznt": "independent",
    "quantigatn": "quantization",
    "quantizal": "quantization",
    "quontiatim": "quantization",
    "elementag": "elementary",
    "exbo": "expo",
    "exbonetih": "exponential",
    "exfonenhals": "exponentials",
    "perod": "period",
    "periodre": "periodic",
    "poriodic": "periodic",
    "peodic": "periodic",
    "þeriod": "period",
    "þevicd": "period",
    "freqinay": "frequency",
    "freqeny": "frequency",
    "phasur": "phasor",
    "origi": "original",
    "posi": "position",
    "rotatton": "rotation",
    "sufficiert": "sufficient",
    "eopb": "complex",
    "seme": "same",
    "expanenial": "exponential",
    "fundanenta": "fundamental",
    "periòd": "period",
    "perndc": "periodic",
}


def fix_common_ocr_errors(text: str) -> str:
    for wrong, correct in COMMON_OCR_FIXES.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", correct, text, flags=re.IGNORECASE)
    return text


def clean_text(text: str) -> str:
    text = fix_common_ocr_errors(text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page\s+\d+", " ", text, flags=re.IGNORECASE)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned_sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 35]

    if len(cleaned_sentences) <= 1:
        words = text.split()
        cleaned_sentences = [
            " ".join(words[start : start + 28]).strip()
            for start in range(0, len(words), 28)
            if len(words[start : start + 28]) >= 8
        ]

    return cleaned_sentences


def split_paragraphs(text: str, max_words: int = 120, overlap: int = 30) -> List[str]:
    """Split text into overlapping word chunks for better RAG retrieval."""
    words = text.split()
    chunks = []
    step = max(1, max_words - overlap)

    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + max_words])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def important_words(text: str) -> List[str]:
    words = re.findall(r"\b[A-Za-z][A-Za-z-]{3,}\b", text.lower())
    return [word for word in words if word not in STOPWORDS]
