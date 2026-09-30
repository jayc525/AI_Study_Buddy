from collections import Counter
from typing import List, Optional

from study_buddy.ollama_client import OllamaClient
from study_buddy.text_utils import important_words, split_sentences


def get_key_sentences(text: str, sentence_count: int = 6):
    """Return important sentences using word frequency scoring."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    frequencies = Counter(important_words(text))
    if not frequencies:
        return []

    scored_sentences = []
    for position, sentence in enumerate(sentences):
        words = important_words(sentence)
        if not words:
            continue

        score = sum(frequencies[word] for word in words) / len(words)
        score += max(0, 1 - position / max(len(sentences), 1)) * 0.15
        scored_sentences.append((score, position, sentence))

    best = sorted(scored_sentences, reverse=True)[:sentence_count]
    best_in_original_order = sorted(best, key=lambda item: item[1])

    return [sentence for _, _, sentence in best_in_original_order]


def summarize_text(text: str, sentence_count: int = 6) -> str:
    """Create a simple extractive summary using word frequency scoring."""
    key_sentences = get_key_sentences(text, sentence_count)
    if not key_sentences:
        return "I could not find enough readable text to summarize."

    return " ".join(key_sentences)


def create_notes_markdown(text: str, point_count: int = 6) -> str:
    """Create Markdown notes with a heading and bullet points (extractive)."""
    key_sentences = get_key_sentences(text, point_count)
    if not key_sentences:
        return "## Notes\n\n- I could not find enough readable text to create notes."

    bullets = [f"- {sentence}" for sentence in key_sentences]
    return "## Notes\n\n" + "\n".join(bullets)


def create_llm_notes(text: str, model_name: str = "phi3", point_count: int = 6) -> Optional[str]:
    """Generate abstractive study notes using the local Ollama LLM.

    Returns None if Ollama is not available, so the caller can fall back
    to the extractive method.
    """
    client = OllamaClient(model_name=model_name)
    if not client.is_available():
        return None

    # Use the most important parts of the text to stay within context limits
    trimmed = text[:3000]

    prompt = f"""You are a study assistant. Create {point_count} clear, concise study notes from the text below.

Rules:
- Write each note as a bullet point starting with "- "
- Rephrase in your own words, do not copy sentences
- Focus on key concepts, definitions, and important facts
- Keep each bullet point to 1-2 sentences
- Do not add a title or heading

Text:
{trimmed}

Study notes:"""

    try:
        response = client.generate(prompt)
        # Validate that we got bullet points back
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        bullet_lines = [line for line in lines if line.startswith("- ") or line.startswith("• ")]

        if len(bullet_lines) >= 2:
            # Normalize bullet style
            normalized = [line.replace("• ", "- ", 1) if line.startswith("• ") else line for line in bullet_lines]
            return "## AI-Generated Notes\n\n" + "\n".join(normalized[:point_count])
        return None
    except Exception:
        return None
