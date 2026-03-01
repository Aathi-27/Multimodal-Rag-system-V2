"""
SLM Engine - Optional small language model for auxiliary tasks.

Handles lightweight inference tasks that don't need the primary LLM:
- Query rewriting (expand/clarify user queries)
- Context compression (summarize retrieved chunks to fit token budget)
- Answer verification (cross-check primary LLM output)

Uses the same llama-cpp-python backend as the primary LLM but with a
smaller, faster model (e.g. Gemma-2-2B, Phi-2).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from llama_cpp import Llama

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


# ── Prompt templates for SLM tasks ───────────────────────────────────────────

_QUERY_REWRITE_PROMPT = """\
<|im_start|>system
You are a search query optimizer. Rewrite the user's question to improve retrieval from a document corpus. Output ONLY the rewritten query, nothing else.<|im_end|>
<|im_start|>user
Original query: {query}<|im_end|>
<|im_start|>assistant
"""

_CONTEXT_COMPRESS_PROMPT = """\
<|im_start|>system
You are a text summarizer. Compress the following context into a shorter version that preserves all key facts. Output ONLY the compressed text.<|im_end|>
<|im_start|>user
{context}<|im_end|>
<|im_start|>assistant
"""

_VERIFICATION_PROMPT = """\
<|im_start|>system
You are a fact-checker. Given the context and an answer, verify if the answer is supported by the context. Reply with "SUPPORTED", "PARTIALLY_SUPPORTED", or "NOT_SUPPORTED" followed by a one-sentence explanation.<|im_end|>
<|im_start|>user
Context:
{context}

Answer to verify:
{answer}<|im_end|>
<|im_start|>assistant
"""


class SLMEngine:
    """
    Optional small language model for query rewrite, context compression, and verification.

    This engine is entirely independent of the primary LLMEngine and loads its own model.
    It is disabled by default and only activates when config.slm.enabled is True and the
    model file exists on disk.
    """

    def __init__(self) -> None:
        self._model: Optional[Llama] = None
        self._settings = get_settings()

    @property
    def is_enabled(self) -> bool:
        """Check if SLM is enabled in config."""
        return getattr(self._settings.slm, "enabled", False)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the SLM GGUF model into memory."""
        if not self.is_enabled:
            logger.info("SLM is disabled in config. Skipping load.")
            return

        model_path = Path(self._settings.slm.model_path)
        if not model_path.exists():
            logger.warning(
                "SLM model not found at %s. SLM features will be unavailable.",
                model_path,
            )
            return

        logger.info("Loading SLM model from %s", model_path)
        self._model = Llama(
            model_path=str(model_path),
            n_ctx=self._settings.slm.context_window,
            n_gpu_layers=self._settings.slm.gpu_layers,
            verbose=False,
        )
        logger.info("SLM model loaded successfully.")

    def unload(self) -> None:
        """Release the SLM from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("SLM model unloaded.")

    def _generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Run a single generation with the SLM."""
        if not self.is_loaded:
            raise RuntimeError("SLM model not loaded.")

        max_tokens = max_tokens or self._settings.slm.max_new_tokens
        output = self._model.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=self._settings.slm.temperature,
            stop=["<|im_end|>", "<|endoftext|>"],
            echo=False,
        )
        return output["choices"][0]["text"].strip()

    # ── Public task methods ──────────────────────────────────────────────────

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite a user query to improve retrieval quality.

        If SLM is not loaded, returns the original query unchanged.
        """
        if not self.is_loaded:
            return query
        try:
            rewritten = self._generate(
                _QUERY_REWRITE_PROMPT.format(query=query),
                max_tokens=128,
            )
            if rewritten and len(rewritten) > 5:
                logger.info("Query rewritten: '%s' → '%s'", query, rewritten)
                return rewritten
            return query
        except Exception as e:
            logger.warning("SLM query rewrite failed: %s", e)
            return query

    def compress_context(self, context: str, max_tokens: int = 200) -> str:
        """
        Compress a context block to fit within a token budget.

        If SLM is not loaded, returns the original context truncated.
        """
        if not self.is_loaded:
            return context
        try:
            compressed = self._generate(
                _CONTEXT_COMPRESS_PROMPT.format(context=context),
                max_tokens=max_tokens,
            )
            if compressed:
                logger.info(
                    "Context compressed: %d chars → %d chars",
                    len(context), len(compressed),
                )
                return compressed
            return context
        except Exception as e:
            logger.warning("SLM context compression failed: %s", e)
            return context

    def verify_answer(self, context: str, answer: str) -> dict:
        """
        Verify that an answer is supported by the given context.

        Returns:
            {"verdict": "SUPPORTED"|"PARTIALLY_SUPPORTED"|"NOT_SUPPORTED",
             "explanation": "..."}

        If SLM is not loaded, returns a default pass-through verdict.
        """
        if not self.is_loaded:
            return {"verdict": "UNCHECKED", "explanation": "SLM not available."}
        try:
            raw = self._generate(
                _VERIFICATION_PROMPT.format(context=context, answer=answer),
                max_tokens=100,
            )
            # Parse the first word as verdict
            parts = raw.split(None, 1)
            verdict = parts[0].upper().rstrip(".:,") if parts else "UNKNOWN"
            explanation = parts[1].strip() if len(parts) > 1 else ""

            valid_verdicts = {"SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"}
            if verdict not in valid_verdicts:
                verdict = "UNKNOWN"

            return {"verdict": verdict, "explanation": explanation}
        except Exception as e:
            logger.warning("SLM verification failed: %s", e)
            return {"verdict": "ERROR", "explanation": str(e)}
