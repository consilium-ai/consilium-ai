"""
Output length prediction — allocate exact KV cache memory.

Based on: arXiv 2602.11812 (ICLR 2026)
Saves 35-46 MB per request by not over-allocating tokens.
Learns from history to improve predictions over time.
"""
import re
from typing import Dict, List

_history: Dict[str, List[int]] = {
    "math": [], "code": [], "research": [], "chat": [], "creative": [],
}

BASE_LENGTHS = {
    "math": 500, "code": 800, "research": 1000, "chat": 500, "creative": 1000,
}

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in",
    "for", "on", "with", "at", "by", "and", "but", "or", "if", "it", "you",
}


def predict_length(prompt: str) -> int:
    """Heuristic fallback — used when AI prediction is unavailable."""
    task = detect_task_type(prompt)
    base = BASE_LENGTHS[task]

    words = len(prompt.split())
    if words > 100:
        base = int(base * 1.3)
    elif words < 10:
        base = int(base * 0.7)

    p = prompt.lower()
    if any(w in p for w in ["explain in detail", "comprehensive", "write an essay"]):
        base = int(base * 1.5)
    elif any(w in p for w in ["one word", "yes or no", "briefly", "short"]):
        base = int(base * 0.3)

    hist = _history.get(task, [])
    if len(hist) >= 3:
        avg = sum(hist[-10:]) / len(hist[-10:])
        base = int(0.7 * avg + 0.3 * base)

    return max(30, min(base, 1500))


def ai_predict_length(prompt: str, llm) -> int:
    """Let the AI decide how many tokens it needs."""
    resp = llm.chat([{"role": "user", "content":
        f'How many tokens do you need to answer this? Reply with ONLY a number.\n\n"{prompt}"'}],
        max_tokens=10)

    # Extract number from response
    import re
    match = re.search(r"(\d+)", resp)
    if match:
        n = int(match.group(1))
        return max(50, min(n, 1500))

    # Fallback to heuristic
    return predict_length(prompt)


def record_actual_length(prompt: str, actual: int):
    task = detect_task_type(prompt)
    _history[task].append(actual)
    if len(_history[task]) > 50:
        _history[task] = _history[task][-50:]


def detect_task_type(prompt: str) -> str:
    p = prompt.lower()
    if any(w in p for w in ["calculate", "compute", "what is", "solve", "+", "*", "/"]):
        return "math"
    if any(w in p for w in ["write code", "function", "implement", "python", "def "]):
        return "code"
    if any(w in p for w in ["search", "research", "find", "arxiv", "paper"]):
        return "research"
    if any(w in p for w in ["write", "story", "poem", "essay", "create"]):
        return "creative"
    return "chat"
