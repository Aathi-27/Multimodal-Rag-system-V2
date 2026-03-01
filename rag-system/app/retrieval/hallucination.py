"""
Hallucination Detector — Answer Grounding Verification.

Checks whether every sentence in the LLM answer can be traced back
to at least one retrieved chunk.  Uses lightweight token overlap
(no extra model needed — keeps it fast on GTX 1650).

Outputs:
  - grounded_ratio    : fraction of answer sentences grounded in context
  - ungrounded_claims : list of sentences that have no context support
  - hallucination_risk: "low" | "medium" | "high"

Business metric:  Directly reduces trust risk and audit liability.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
LOW_RISK_THRESHOLD = 0.85      # ≥85% grounded → low risk
MEDIUM_RISK_THRESHOLD = 0.60   # ≥60% grounded → medium risk
MIN_OVERLAP_TOKENS = 3         # Minimum shared tokens for a sentence to be "grounded"
MIN_OVERLAP_RATIO = 0.30       # ≥30% of sentence tokens must appear in context


@dataclass
class HallucinationResult:
    """Grounding assessment for a single answer."""
    grounded_ratio: float           # 0.0–1.0
    risk_level: str                 # "low" | "medium" | "high"
    total_sentences: int
    grounded_sentences: int
    ungrounded_claims: list[str]    # Sentences with no context support

    def to_dict(self) -> dict:
        return {
            "grounded_ratio": round(self.grounded_ratio, 3),
            "risk_level": self.risk_level,
            "total_sentences": self.total_sentences,
            "grounded_sentences": self.grounded_sentences,
            "ungrounded_claims": self.ungrounded_claims[:5],  # Cap for payload size
        }


# ── Tokenizer (lightweight) ──────────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "our", "their", "and", "or", "but", "nor", "not", "so", "yet",
    "for", "of", "in", "on", "at", "to", "from", "by", "with", "as",
    "if", "then", "than", "also", "just", "very", "more", "most",
    "such", "only", "own", "same", "other", "each", "every",
})

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _tokenize(text: str) -> set[str]:
    """Extract content words (lowercase, stopwords removed)."""
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering trivial fragments."""
    raw = _SENT_SPLIT.split(text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 15]


# ── Core detection ────────────────────────────────────────────────────────────

def detect_hallucination(
    answer: str,
    chunk_texts: list[str],
) -> HallucinationResult:
    """
    Check how well the answer is grounded in the retrieved context.

    Args:
        answer:      The generated LLM answer text.
        chunk_texts: List of plain text from the used context chunks.

    Returns:
        HallucinationResult with grounding ratio and ungrounded claims.
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return HallucinationResult(
            grounded_ratio=1.0, risk_level="low",
            total_sentences=0, grounded_sentences=0,
            ungrounded_claims=[],
        )

    # Build a combined context token set (fast lookup)
    context_tokens: set[str] = set()
    for chunk in chunk_texts:
        context_tokens.update(_tokenize(chunk))

    grounded = 0
    ungrounded_claims: list[str] = []

    for sent in sentences:
        sent_tokens = _tokenize(sent)
        if not sent_tokens:
            grounded += 1  # Trivial sentence (no content words) — skip
            continue

        overlap = sent_tokens & context_tokens
        overlap_ratio = len(overlap) / len(sent_tokens)

        if len(overlap) >= MIN_OVERLAP_TOKENS and overlap_ratio >= MIN_OVERLAP_RATIO:
            grounded += 1
        else:
            ungrounded_claims.append(sent)

    ratio = grounded / len(sentences) if sentences else 1.0

    if ratio >= LOW_RISK_THRESHOLD:
        risk = "low"
    elif ratio >= MEDIUM_RISK_THRESHOLD:
        risk = "medium"
    else:
        risk = "high"

    return HallucinationResult(
        grounded_ratio=ratio,
        risk_level=risk,
        total_sentences=len(sentences),
        grounded_sentences=grounded,
        ungrounded_claims=ungrounded_claims,
    )
