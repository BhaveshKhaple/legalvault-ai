"""
Task 4.1 — Ollama wrapper.

Ollama runs open-source LLMs (Phi-4, Qwen3, Llama 3.2) locally via an HTTP
API at localhost:11434. Data never leaves the machine.

This module provides a single generate() function so the rest of the stack
doesn't know or care about Ollama's HTTP protocol.

Resilience:
    - Requests that exceed OLLAMA_TIMEOUT_SEC raise a clear TimeoutError.
    - If the Ollama server is not running, raises OllamaUnavailable (not a
      raw ConnectionError) with an actionable message.

Usage:
    from backend.app.llm.ollama_client import generate

    answer = generate(
        prompt="What are the penalty clauses?",
        context=["Clause 8.3: penalty interest at 2% per month..."],
    )
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

# ─── config ───────────────────────────────────────────────────────────────────

_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "phi4")
_TIMEOUT_SEC = int(os.environ.get("OLLAMA_TIMEOUT_SEC", "60"))

_GENERATE_URL = f"{_OLLAMA_HOST}/api/generate"


# ─── exceptions ───────────────────────────────────────────────────────────────


class OllamaUnavailable(RuntimeError):
    """Raised when the Ollama server cannot be reached."""


# ─── public API ───────────────────────────────────────────────────────────────


def generate(
    prompt: str,
    context: list[str] | None = None,
    model: str = _DEFAULT_MODEL,
    timeout: int = _TIMEOUT_SEC,
) -> str:
    """Send a prompt to Ollama and return the generated text.

    Args:
        prompt:  The user question or instruction.
        context: Optional list of retrieved chunk texts to include as
                 grounding context. Joined with newlines and prepended to
                 the prompt in a simple RAG pattern. Task 4.3 will replace
                 this with a proper citation-enforcing prompt template.
        model:   Ollama model name. Defaults to OLLAMA_MODEL env var or 'phi4'.
        timeout: Per-request timeout in seconds. Defaults to OLLAMA_TIMEOUT_SEC.

    Returns:
        The model's response as a plain string.

    Raises:
        OllamaUnavailable: If the Ollama server is not running or unreachable.
        TimeoutError: If the request exceeds ``timeout`` seconds.
    """
    full_prompt = _build_prompt(prompt, context or [])

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
    }

    logger.debug("Ollama generate — model='%s', prompt_len=%d chars.", model, len(full_prompt))

    try:
        response = requests.post(
            _GENERATE_URL,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise OllamaUnavailable(
            f"Cannot reach Ollama at '{_OLLAMA_HOST}'. "
            "Make sure Ollama is installed and running: 'ollama serve'."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise TimeoutError(
            f"Ollama request timed out after {timeout}s. "
            "Try a smaller model (MODEL_TIER=3) or increase OLLAMA_TIMEOUT_SEC."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise OllamaUnavailable(
            f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text[:200]}"
        ) from exc

    data = response.json()
    text = data.get("response", "").strip()

    logger.info(
        "Ollama generate complete — model='%s', response_len=%d chars.",
        model,
        len(text),
    )
    return text


# ─── helpers ──────────────────────────────────────────────────────────────────


def _build_prompt(question: str, context_chunks: list[str]) -> str:
    """Assemble context + question into a single prompt string.

    Task 4.3 will replace this with the citation-enforcing prompt template.
    This placeholder is intentionally minimal.
    """
    if not context_chunks:
        return question

    context_block = "\n\n".join(
        f"[Document excerpt {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    return (
        f"Use only the following document excerpts to answer the question. "
        f"Do not use prior knowledge.\n\n"
        f"{context_block}\n\n"
        f"Question: {question}"
    )
