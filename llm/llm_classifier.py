"""
llm/llm_classifier.py
---------------------
Stage 1 of the multi-stage pipeline.

Uses a fast, deterministic local LLM (mistral:7b) to classify any user
input — URL, pasted text, or OCR string — into a known source_type.

Design:
  - temperature=0.0  → fully deterministic, same input = same output
  - num_predict=120  → only needs ~50 tokens for the JSON response
  - mistral:7b       → fast, ~1–2s on RTX 3050, good at instruction following
  - Prompt lives in  → llm/prompt_builder.py (build_classifier_prompt)

Does NOT:
  - Call processors       → processors/*.py
  - Clean text            → utils/cleaner.py
  - Build final answers   → llm/summarizer.py
  - Run full pipeline     → llm/pipeline.py

Public API:
  classify(user_input)  → dict with source_type, confidence, reason
  is_supported(source_type) → bool

Called by:
  llm/pipeline.py     → classify()
  main.py             → _detect_source_smart() fallback
"""

from __future__ import annotations

import json
import logging
import requests

from llm.prompt_builder import build_classifier_prompt

# Try to read Ollama URL from config; fall back to default
try:
    from config import OLLAMA_URL  # type: ignore
except ImportError:
    OLLAMA_URL = "http://localhost:11434/api/generate"

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIG
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIFIER_MODEL = "mistral:7b"

_CLASSIFIER_OPTIONS = {
    "temperature": 0.0,    # deterministic — no randomness for classification
    "num_predict": 120,    # only need ~50 tokens for the JSON response
    "top_p":       1.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# KNOWN SOURCE TYPES
# ─────────────────────────────────────────────────────────────────────────────

# All valid source types the classifier may return.
# Anything outside this set is treated as "plain_text" (safe fallback).
_VALID_SOURCE_TYPES: set[str] = {
    # YouTube
    "youtube_video", "youtube_shorts", "youtube_playlist", "youtube_channel",
    # GitHub
    "github_repo", "github_user", "github_file", "github_gist",
    # Web / Publishing
    "web", "medium_article", "substack_article", "notion_page",
    "reddit_post", "reddit_subreddit",
    # Research
    "arxiv_paper", "pdf_document",
    # Social
    "instagram_post", "instagram_reel", "instagram_profile",
    "linkedin_profile", "linkedin_post", "linkedin_company",
    # AI / ML
    "huggingface_model", "huggingface_dataset", "huggingface_space",
    # Video
    "loom_video", "vimeo_video",
    # Local / Raw
    "local_image", "code_snippet", "notebook",
    "plain_text", "news_headline", "json_data", "markdown",
    # Blocked
    "unsupported",
}

# Source types that main.py handles with a friendly refusal message.
# Classifier should return these so main.py can intercept them early.
_UNSUPPORTED_SOURCE_TYPES: set[str] = {
    "instagram_profile",
    "linkedin_profile",
    "linkedin_company",
    "github_user",
    "unsupported",
}


# ─────────────────────────────────────────────────────────────────────────────
# JSON EXTRACTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """
    Safely extract a JSON object from LLM response text.
    Handles cases where the model adds markdown fences or explanation text.
    """
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last fence lines
        text = "\n".join(
            l for l in lines
            if not l.strip().startswith("```")
        )

    start = text.find("{")
    end   = text.rfind("}") + 1

    if start == -1 or end == 0 or end <= start:
        logger.warning(f"[CLASSIFIER] No JSON object found in response: {text[:100]}")
        return {}

    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError as e:
        logger.warning(f"[CLASSIFIER] JSON parse error: {e} | text: {text[start:end][:100]}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# CORE LLM CALL
# ─────────────────────────────────────────────────────────────────────────────

def _call_classifier(prompt: str) -> str:
    """
    Raw HTTP call to Ollama using the classifier model.
    Separate from summarizer.call_llm() to use different model + options.
    """
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model":   _CLASSIFIER_MODEL,
                "prompt":  prompt,
                "stream":  False,
                "options": _CLASSIFIER_OPTIONS,
            },
            timeout=30,    # classifier must be fast — 30s max
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()

    except requests.exceptions.ConnectionError:
        logger.error("[CLASSIFIER] Ollama not running")
        return ""
    except requests.exceptions.Timeout:
        logger.error(f"[CLASSIFIER] Timed out after 30s (model: {_CLASSIFIER_MODEL})")
        return ""
    except requests.exceptions.HTTPError as e:
        if "404" in str(e):
            logger.error(
                f"[CLASSIFIER] Model '{_CLASSIFIER_MODEL}' not found. "
                f"Run: ollama pull {_CLASSIFIER_MODEL}"
            )
        else:
            logger.error(f"[CLASSIFIER] HTTP error: {e}")
        return ""
    except Exception as e:
        logger.error(f"[CLASSIFIER] Unexpected error: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def classify(user_input: str) -> dict:
    """
    Classify user input into a source_type using a fast LLM call.

    Args:
        user_input: Any string — URL, pasted text, OCR output, headline, etc.
                    Truncated to 800 chars internally (prompt_builder handles this).

    Returns:
        dict with keys:
          source_type  : str  — one of _VALID_SOURCE_TYPES
          confidence   : str  — "high" | "medium" | "low"
          reason       : str  — one sentence explaining the classification
          raw          : str  — original LLM response (for debugging)

    Never raises — always returns a safe fallback dict on any failure.

    Example:
        result = classify("https://github.com/openai/whisper")
        # → {"source_type": "github_repo", "confidence": "high", "reason": "..."}
    """
    if not user_input or not user_input.strip():
        return _fallback("empty input")

    prompt  = build_classifier_prompt(user_input)
    raw     = _call_classifier(prompt)

    if not raw:
        return _fallback("LLM returned empty response")

    parsed = _extract_json(raw)

    if not parsed:
        return _fallback(f"could not parse JSON from: {raw[:80]}")

    # Validate and sanitize source_type
    source_type = parsed.get("source_type", "").strip().lower()
    if source_type not in _VALID_SOURCE_TYPES:
        logger.warning(
            f"[CLASSIFIER] Unknown source_type '{source_type}', "
            f"falling back to plain_text"
        )
        source_type = "plain_text"

    confidence = parsed.get("confidence", "low").strip().lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    result = {
        "source_type": source_type,
        "confidence":  confidence,
        "reason":      parsed.get("reason", "").strip(),
        "raw":         raw,
    }

    logger.info(
        f"[CLASSIFIER] {source_type!r} "
        f"(confidence={confidence}, reason={result['reason']})"
    )
    print(
        f"  [S1-CLASSIFY] {source_type!r} "
        f"({confidence}) — {result['reason']}"
    )

    return result


def is_supported(source_type: str) -> bool:
    """
    Returns False if source_type is one that main.py cannot process
    (login-required, JS-only, user profile pages, etc.).

    Usage:
        clf = classify(url)
        if not is_supported(clf["source_type"]):
            return friendly_refusal_message
    """
    return source_type not in _UNSUPPORTED_SOURCE_TYPES


def _fallback(reason: str) -> dict:
    """Return a safe default classification result."""
    logger.warning(f"[CLASSIFIER] Fallback to plain_text — {reason}")
    return {
        "source_type": "plain_text",
        "confidence":  "low",
        "reason":      f"Classification failed ({reason}), defaulting to plain_text",
        "raw":         "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI TEST  (python -m llm.llm_classifier)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tests = [
        # URLs
        ("https://youtube.com/watch?v=abc123",              "youtube_video"),
        ("https://github.com/openai/whisper",               "github_repo"),
        ("https://github.com/palinkiewicz",                  "github_user"),
        ("https://arxiv.org/abs/2303.08774",                "arxiv_paper"),
        ("https://huggingface.co/mistralai/Mistral-7B-v0.1","huggingface_model"),
        ("https://www.instagram.com/aditya.ksx/",           "instagram_profile"),
        ("https://linkedin.com/in/johndoe",                  "linkedin_profile"),
        ("https://adityaksx.vercel.app/",                   "unsupported"),
        ("https://medium.com/@user/my-article",             "medium_article"),
        ("https://reddit.com/r/Python/comments/abc",        "reddit_post"),
        # Raw text
        ("def fibonacci(n): return n if n < 2 else ...",    "code_snippet"),
        ('{"name": "aditya", "role": "dev"}',               "json_data"),
        ("# My Notes\n- Point 1\n- Point 2\n**bold**",      "markdown"),
        ("India's Sarvam launches Indus AI chat app",        "news_headline"),
        ("FastAPI is a fast Python web framework.",          "plain_text"),
    ]

    print(f"\n{'Input':<52} {'Expected':<22} {'Got':<22} {'Conf':<8} {'✓/?'}")
    print("─" * 115)

    passed = 0
    failed = 0

    for inp, expected in tests:
        result      = classify(inp)
        got         = result["source_type"]
        conf        = result["confidence"]
        ok          = "✅" if got == expected else "❌"
        if got == expected:
            passed += 1
        else:
            failed += 1
        print(f"{inp[:51]:<52} {expected:<22} {got:<22} {conf:<8} {ok}")

    print(f"\n{passed}/{passed + failed} passed")
