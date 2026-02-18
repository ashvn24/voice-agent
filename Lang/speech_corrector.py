"""
Vidur AI — Speech Autocorrect
Fixes misheard technical terms from Google STT.
Add new corrections to CORRECTIONS dict as you discover them.
"""
import re
from difflib import SequenceMatcher


CORRECTIONS = {
    # ── AI Frameworks ──
    "land chain": "LangChain",   "lang chain": "LangChain",
    "long chain": "LangChain",   "lung chain": "LangChain",
    "lank chain": "LangChain",   "line chain": "LangChain",
    "lang graph": "LangGraph",   "long graph": "LangGraph",
    "lung graph": "LangGraph",
    "llama index": "LlamaIndex", "lama index": "LlamaIndex",
    "crew ai": "CrewAI",         "crew a i": "CrewAI",
    "auto gen": "AutoGen",       "autogen": "AutoGen",

    # ── AI / ML ──
    "chat gpt": "ChatGPT",      "chat g p t": "ChatGPT",
    "chat gbt": "ChatGPT",
    "open ai": "OpenAI",         "open a i": "OpenAI",
    "hugging face": "HuggingFace", "hugging phase": "HuggingFace",
    "gemini": "Gemini",          "mistral": "Mistral",
    "gpt 4": "GPT-4",           "gpt for": "GPT-4",
    "r a g": "RAG",
    "pine cone": "Pinecone",     "chroma db": "ChromaDB",
    "pie torch": "PyTorch",      "pytorch": "PyTorch",
    "tensor flow": "TensorFlow",
    "fine tune": "fine-tuning",  "fine tuning": "fine-tuning",

    # ── Programming ──
    "java script": "JavaScript", "react js": "ReactJS",
    "node js": "NodeJS",         "next js": "NextJS",
    "fast api": "FastAPI",       "jango": "Django",
    "cooper netties": "Kubernetes",
    "git hub": "GitHub",         "vs code": "VS Code",

    # ── Education ──
    "you dummy": "Udemy",        "coursera": "Coursera",
    "fafsa": "FAFSA",
}


def correct(text: str) -> str:
    """Exact-match correction of misheard terms."""
    corrected = text
    for wrong, right in CORRECTIONS.items():
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        if pattern.search(corrected):
            corrected = pattern.sub(right, corrected)

    if corrected != text:
        print(f"  🔧 Autocorrected: \"{text}\" → \"{corrected}\"")
    return corrected


def fuzzy_correct(text: str, threshold: float = 0.75) -> str:
    """Fuzzy match multi-word chunks against known misheard terms."""
    words = text.lower().split()
    for n in [2, 3]:
        for i in range(len(words) - n + 1):
            chunk = " ".join(words[i:i + n])
            for wrong, right in CORRECTIONS.items():
                if SequenceMatcher(None, chunk, wrong).ratio() >= threshold:
                    original_chunk = " ".join(text.split()[i:i + n])
                    text = text.replace(original_chunk, right, 1)
                    print(f"  🔧 Fuzzy corrected: \"{chunk}\" → \"{right}\"")
                    return text
    return text


def auto_correct(text: str) -> str:
    """Run both exact and fuzzy correction."""
    text = correct(text)
    text = fuzzy_correct(text)
    return text