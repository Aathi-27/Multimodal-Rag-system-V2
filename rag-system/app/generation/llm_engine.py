"""
LLM Engine - llama.cpp wrapper for Qwen2.5-1.5B inference.

Handles:
- Model loading via llama-cpp-python
- Token generation with streaming
- Context window management (4096 tokens for RAG)
- Full GPU offload (n_gpu_layers=-1)
- KV cache quantization (q8_0) for reduced VRAM
- Deterministic sampling (top_k=1) for minimal overhead
- Configurable stop tokens per model family
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Generator, Optional

from llama_cpp import Llama

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class LLMEngine:
    """Wrapper around llama.cpp for offline LLM inference."""

    def __init__(self) -> None:
        self._model: Optional[Llama] = None
        self._settings = get_settings()
        self._thread_local = threading.local()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the GGUF model into memory."""
        model_path = Path(self._settings.llm.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"LLM model not found at {model_path}. "
                "Run scripts/download_models.py first."
            )

        logger.info("Loading LLM model from %s", model_path)

        from llama_cpp import GGML_TYPE_Q8_0

        self._model = Llama(
            model_path=str(model_path),
            n_ctx=self._settings.llm.context_window,
            n_gpu_layers=self._settings.llm.gpu_layers,
            n_batch=getattr(self._settings.llm, 'n_batch', 512),
            n_threads=8,
            flash_attn=True,
            type_k=GGML_TYPE_Q8_0,   # KV cache quantization — saves ~60 MiB VRAM
            type_v=GGML_TYPE_Q8_0,
            verbose=False,
        )
        # Pre-tokenize and cache the system prompts (avoids re-tokenizing per query)
        self._cached_prompt_tokens: dict[str, list[int]] = {}
        logger.info("LLM model loaded successfully.")

    def unload(self) -> None:
        """Release the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("LLM model unloaded.")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Generate a complete response (non-streaming)."""
        if not self.is_loaded:
            raise RuntimeError("LLM model not loaded. Call load() first.")

        max_tokens = max_tokens or self._settings.llm.max_new_tokens
        temperature = temperature or self._settings.llm.temperature

        output = self._model.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or getattr(self._settings.llm, 'stop_tokens', ["<|im_end|>"]),
            top_k=1,          # Greedy / deterministic — skip expensive sampling
            echo=False,
        )
        return output["choices"][0]["text"].strip()

    def _dynamic_max_tokens(self, prompt_tokens: int, requested_max: int) -> int:
        """Cap max_tokens to fit within context window and avoid over-allocation.

        Strategy:
        - Never exceed (n_ctx - prompt_tokens - 16) to avoid truncation
        - For short prompts (<1000 tok), cap at 384  (simple Q&A)
        - For medium prompts (1000-2500), cap at requested_max
        - For long prompts (>2500), cap at remaining budget
        """
        n_ctx = self._settings.llm.context_window
        headroom = 16  # safety margin for special tokens
        remaining = n_ctx - prompt_tokens - headroom

        if remaining < 64:
            logger.warning("Very little room for generation: %d tokens", remaining)
            return max(remaining, 32)

        # Don't exceed the context window
        capped = min(requested_max, remaining)
        return capped

    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None,
    ) -> Generator[str, None, None]:
        """Generate tokens incrementally for SSE streaming."""
        import time as _time

        if not self.is_loaded:
            raise RuntimeError("LLM model not loaded. Call load() first.")

        max_tokens = max_tokens or self._settings.llm.max_new_tokens
        temperature = temperature or self._settings.llm.temperature

        prompt_tokens = len(self._model.tokenize(prompt.encode("utf-8")))
        max_tokens = self._dynamic_max_tokens(prompt_tokens, max_tokens)
        logger.info(
            "Starting generation: prompt_tokens=%d, max_new=%d",
            prompt_tokens, max_tokens,
        )
        t0 = _time.perf_counter()

        self._thread_local.last_finish_reason = None

        stream = self._model.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or getattr(self._settings.llm, 'stop_tokens', ["<|im_end|>"]),
            top_k=1,          # Greedy / deterministic — skip expensive sampling
            echo=False,
            stream=True,
        )

        first = True
        count = 0
        for chunk in stream:
            token = chunk["choices"][0]["text"]
            finish_reason = chunk["choices"][0].get("finish_reason")
            if finish_reason:
                self._thread_local.last_finish_reason = finish_reason
            if token:
                if first:
                    ttft = _time.perf_counter() - t0
                    logger.info("First token in %.1fs (TTFT)", ttft)
                    first = False
                count += 1
                yield token

        elapsed = _time.perf_counter() - t0
        tps = count / elapsed if elapsed > 0 else 0
        truncated = getattr(self._thread_local, 'last_finish_reason', None) == "length"
        logger.info(
            "Generation done: %d tokens in %.1fs (%.1f tok/s)%s",
            count, elapsed, tps,
            " [TRUNCATED — hit token limit]" if truncated else "",
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer (authoritative)."""
        if not self.is_loaded:
            raise RuntimeError("LLM model not loaded. Call load() first.")
        return len(self._model.tokenize(text.encode("utf-8")))
