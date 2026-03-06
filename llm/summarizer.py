"""
llm/summarizer.py
-----------------
Calls the local Ollama LLM and returns structured knowledge output.

Responsibilities (this file ONLY):
  - Model selection by source_type
  - Raw HTTP call to Ollama API
  - Paragraph-aware chunking for long content
  - Multi-chunk merge via build_merge_prompt()

Does NOT:
  - Build prompts          → llm/prompt_builder.py
  - Run multi-stage flow   → llm/pipeline.py
  - Classify input type    → llm/llm_classifier.py
  - Clean text             → utils/cleaner.py

Public API consumed by other files:
  main.py      → call_llm(), summarize_data()
  pipeline.py  → call_llm()
"""

from __future__ import annotations

import logging
import requests

from llm.prompt_builder import build_summary_prompt, build_merge_prompt

# Try to read Ollama URL from config; fall back to default
try:
    from config import OLLAMA_URL  # type: ignore
except ImportError:
    OLLAMA_URL = "http://localhost:11434/api/generate"

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ROUTING
# ─────────────────────────────────────────────────────────────────────────────

# Code-focused sources → deepseek-coder for better code understanding
_CODE_TYPES: set[str] = {
    "github_repo",
    "github_file",
    "github_gist",    # FIX: was missing — gists are raw code
    "code_snippet",
    "notebook",       # FIX: was missing — Jupyter notebooks are code
}

# Long, dense content → larger model for deeper reasoning
_DEEP_ANALYSIS_TYPES: set[str] = {
    "arxiv_paper",
    "pdf_document",
    "substack_article",
    "medium_article",  # often long-form technical writing
}

_DEFAULT_MODEL = "qwen2.5:7b-instruct"
_CODE_MODEL    = "deepseek-coder:6.7b"
_DEEP_MODEL    = "qwen2.5:14b"

# Per-model generation options
_MODEL_OPTIONS: dict[str, dict] = {
    _CODE_MODEL:    {"temperature": 0.2, "num_predict": 1200, "top_p": 0.9},
    _DEEP_MODEL:    {"temperature": 0.3, "num_predict": 2000, "top_p": 0.9},
    _DEFAULT_MODEL: {"temperature": 0.3, "num_predict": 1500, "top_p": 0.9},
}

_DEFAULT_OPTIONS: dict = {"temperature": 0.3, "num_predict": 1500, "top_p": 0.9}


def get_model(source_type: str) -> str:
    """
    Select the best Ollama model for a given source_type.
    Called by call_llm() automatically — callers don't need to specify model.
    """
    if source_type in _CODE_TYPES:
        return _CODE_MODEL
    if source_type in _DEEP_ANALYSIS_TYPES:
        return _DEEP_MODEL
    return _DEFAULT_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_ollama() -> tuple[bool, str]:
    """
    Check whether Ollama is running and reachable.
    Returns (is_running: bool, message: str).

    Usage:
        ok, msg = check_ollama()
        if not ok: return msg
    """
    try:
        base_url = OLLAMA_URL.replace("/api/generate", "")
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return True, f"Ollama running. Available models: {models}"
        return False, f"Ollama returned status {r.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Ollama is not running. Start it with: ollama serve"
    except Exception as e:
        return False, f"Ollama health check failed: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# CORE LLM CALL
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(prompt: str, source_type: str = "") -> str:
    """
    Send a ready-built prompt string to Ollama.
    Model is selected automatically from source_type.

    Called by:
      - summarize_data()    (Stage 4 final answer)
      - pipeline.py         (Stage 1, 2, 3 intermediate calls)
      - main.py             (multi-input merge)

    Args:
        prompt:      Complete prompt string — no further wrapping done here.
        source_type: Used only for model selection. Empty → default model.

    Returns:
        LLM response string, or an [ERROR] prefixed message on failure.
    """
    model   = get_model(source_type)
    options = _MODEL_OPTIONS.get(model, _DEFAULT_OPTIONS)

    logger.info(f"[LLM] model={model}  source={source_type or 'default'}")
    print(f"  [LLM] model={model}  source={source_type or 'default'}")

    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model":   model,
                "prompt":  prompt,
                "stream":  False,
                "options": options,
            },
            timeout=240,
        )
        r.raise_for_status()
        response = r.json().get("response", "").strip()

        if not response:
            logger.warning(f"[LLM] Empty response from {model}")
            return "[ERROR] LLM returned an empty response. Try again."

        return response

    except requests.exceptions.ConnectionError:
        msg = "[ERROR] Ollama is not running. Start it with: ollama serve"
        logger.error(msg)
        return msg
    except requests.exceptions.Timeout:
        msg = f"[ERROR] Ollama timed out after 240s (model: {model})."
        logger.error(msg)
        return msg
    except requests.exceptions.HTTPError as e:
        # 404 usually means the model isn't pulled yet
        if "404" in str(e):
            msg = (
                f"[ERROR] Model '{model}' not found in Ollama.\n"
                f"Run: ollama pull {model}"
            )
        else:
            msg = f"[ERROR] Ollama HTTP error: {e}"
        logger.error(msg)
        return msg
    except Exception as e:
        msg = f"[ERROR] LLM call failed: {e}"
        logger.error(msg)
        return msg


# ─────────────────────────────────────────────────────────────────────────────
# PARAGRAPH-AWARE CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, max_chars: int = 3000) -> list[str]:
    """
    Split text into chunks ≤ max_chars, breaking at paragraph then sentence
    boundaries. Never cuts mid-sentence.

    Used by summarize_data() when a field exceeds 3000 characters.
    """
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text]

    chunks:  list[str] = []
    current: str       = ""

    for para in paragraphs:
        # Paragraph fits in current chunk
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para).strip() if current else para

        else:
            # Save current chunk before starting a new one
            if current:
                chunks.append(current)
                current = ""

            # Paragraph itself is too long — split at sentence level
            if len(para) > max_chars:
                sentences = para.replace(". ", ".\n").split("\n")
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if len(current) + len(sent) + 1 <= max_chars:
                        current = (current + " " + sent).strip() if current else sent
                    else:
                        # FIX: flush current before starting next sentence
                        if current:
                            chunks.append(current)
                        # If single sentence exceeds max_chars, keep it as-is
                        current = sent
            else:
                current = para

    # Flush remaining content
    if current:
        chunks.append(current)

    return chunks if chunks else [text]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SUMMARIZATION ENTRY POINTS
# ─────────────────────────────────────────────────────────────────────────────

def summarize_data(data: dict) -> str:
    """
    Main entry point. Accepts a cleaned + enriched processor output dict.
    Builds prompt via prompt_builder.py and calls the correct model.
    Handles chunking for long fields automatically.

    Called by: main.py → _run_pipeline()
    Receives:  already-cleaned dict from utils/cleaner.py
               already-enriched dict from llm/pipeline.py Stage 3

    Args:
        data: Dict with keys like source_type, title, content,
              transcript, readme, ocr_text, description, etc.

    Returns:
        Final structured LLM output string (MAIN IDEA / KEY INSIGHTS / TAGS…)
    """
    source_type = data.get("source_type", "")

    # Fields that might be too long for a single LLM context window
    _LONG_FIELDS = ["transcript", "content", "body", "text", "readme", "ocr_text"]

    needs_chunking = any(
        isinstance(data.get(f), str) and len(data.get(f, "")) > 3000
        for f in _LONG_FIELDS
    )

    # ── Short enough — single call ────────────────────────────────────────
    if not needs_chunking:
        prompt = build_summary_prompt(data)
        return call_llm(prompt, source_type)

    # ── Find the longest field to chunk ──────────────────────────────────
    target_field = max(
        (f for f in _LONG_FIELDS if isinstance(data.get(f), str) and data.get(f)),
        key=lambda f: len(data.get(f, "")),
        default=None,
    )

    if not target_field:
        # No long string field found despite check — just call normally
        return call_llm(build_summary_prompt(data), source_type)

    # ── Chunk and summarize each part ────────────────────────────────────
    chunks = chunk_text(data[target_field])
    logger.info(f"[LLM] Chunking '{target_field}' into {len(chunks)} parts")
    print(f"  [LLM] Chunking '{target_field}' into {len(chunks)} parts")

    partial_summaries: list[str] = []
    for i, chunk in enumerate(chunks):
        print(f"  [LLM] Summarizing chunk {i + 1}/{len(chunks)}...")
        chunk_data = {**data, target_field: chunk}
        partial = call_llm(build_summary_prompt(chunk_data), source_type)
        # Don't accumulate error messages as summaries
        if not partial.startswith("[ERROR]"):
            partial_summaries.append(partial)

    if not partial_summaries:
        return "[ERROR] All chunk summaries failed."

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    # ── Merge partial summaries into one final answer ─────────────────────
    print("  [LLM] Merging chunks into final answer...")
    return call_llm(build_merge_prompt(partial_summaries), source_type="")


def summarize_text(text: str, source_type: str = "plain_text") -> str:
    """
    Convenience wrapper for raw string input (no processor dict needed).

    Usage:
        result = summarize_text("def foo(): ...", source_type="code_snippet")
    """
    if not text or not text.strip():
        return "No text provided."
    return summarize_data({"source_type": source_type, "content": text})


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY ALIAS
# ─────────────────────────────────────────────────────────────────────────────

def summarize(prompt: str) -> str:
    """
    Legacy alias — kept for backwards compatibility only.
    Treats input as a ready-built prompt and sends directly to LLM.
    Does NOT wrap it in another prompt template.

    Prefer call_llm() for new code.
    """
    return call_llm(prompt, source_type="")


# ─────────────────────────────────────────────────────────────────────────────
# CLI  (python -m llm.summarizer)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Checking Ollama...")
    ok, msg = check_ollama()
    print(f"  {'✅' if ok else '❌'} {msg}\n")

    if ok:
        print("Quick test — summarizing plain text:")
        result = summarize_text(
            "FastAPI is a modern Python web framework for building APIs. "
            "It uses type hints for validation and generates OpenAPI docs automatically.",
            source_type="plain_text"
        )
        print(result)
