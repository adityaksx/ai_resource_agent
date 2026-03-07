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
  classify(user_input)      → dict with source_type, confidence, reason
  is_supported(source_type) → bool

Called by:
  llm/pipeline.py     → classify()
  main.py             → _detect_source_smart() fallback
"""

from __future__ import annotations

import asyncio
import json
import logging
import httpx

from llm.prompt_builder  import build_classifier_prompt
from llm.summarizer      import _ollama_semaphore          # ← share the SAME semaphore

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
    "num_predict": 200,    # only need ~50 tokens for the JSON response
    "top_p":       1.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# KNOWN SOURCE TYPES
# ─────────────────────────────────────────────────────────────────────────────

_VALID_SOURCE_TYPES: set[str] = {
    "youtube_video", "youtube_shorts", "youtube_playlist", "youtube_channel",
    "github_repo", "github_user", "github_file", "github_gist",
    "web", "medium_article", "substack_article", "notion_page",
    "reddit_post", "reddit_subreddit",
    "arxiv_paper", "pdf_document",
    "instagram_post", "instagram_reel", "instagram_profile",
    "linkedin_profile", "linkedin_post", "linkedin_company",
    "huggingface_model", "huggingface_dataset", "huggingface_space",
    "loom_video", "vimeo_video",
    "local_image", "code_snippet", "notebook",
    "plain_text", "news_headline", "json_data", "markdown",
    "unsupported",
}

_UNSUPPORTED_SOURCE_TYPES: set[str] = {
    "instagram_profile",
    "linkedin_profile",
    "linkedin_company",
    "github_user",
    "unsupported",
}


# ─────────────────────────────────────────────────────────────────────────────
# JSON EXTRACTION HELPER  (unchanged — pure string logic, no async needed)
# ─────────────────────────────────────────────────────────────────────────────

# REPLACE WITH this brace-walking version (same as pipeline.py):
def _extract_json(text: str) -> dict:
    """
    Safely extract the FIRST complete JSON object from LLM response text.

    Handles:
      - Markdown fences (```json ... ```)
      - Explanation text before or after the JSON
      - Trailing commas
      - Literal newlines inside string values
      - Multiple JSON objects — takes the FIRST complete one
    """
    import re as _re

    if not text:
        return {}

    text = text.strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(l for l in lines if not l.strip().startswith("```"))

    # Find first opening brace
    start = text.find("{")
    if start == -1:
        logger.warning(f"[CLASSIFIER] No JSON found in: {text[:100]}")
        return {}

    # ── Walk braces to find the FIRST complete matching } ────────────────
    depth     = 0
    end       = -1
    in_string = False
    escape    = False

    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        logger.warning(f"[CLASSIFIER] Unmatched braces in: {text[:100]}")
        return {}

    raw_json = text[start:end]

    # Sanitize literal newlines/control chars inside quoted strings
    def _fix_string_newlines(m):
        return m.group(0).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    raw_json = _re.sub(
        r'"[^"\\]*(?:\\.[^"\\]*)*"',
        _fix_string_newlines,
        raw_json,
        flags=_re.DOTALL,
    )
    raw_json = _re.sub(r'(?<!\\)[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', raw_json)

    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        # Try fixing trailing commas as last resort
        fixed = _re.sub(r",\s*([}\]])", r"\1", raw_json)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            logger.warning(f"[CLASSIFIER] JSON parse failed: {e} | raw: {raw_json[:100]}")
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# CORE LLM CALL  (now async + semaphore + retry)
# ─────────────────────────────────────────────────────────────────────────────

async def _call_classifier(prompt: str) -> str:
    """
    Async HTTP call to Ollama using the classifier model.
    Uses the SAME _ollama_semaphore from summarizer.py — ensures
    all Ollama calls across the entire app are serialised.
    """
    payload = {
        "model":  _CLASSIFIER_MODEL,
        "system": (
            "You are a URL and content classifier. "
            "Return ONLY a valid JSON object. "
            "No explanation. No markdown. No extra text before or after the JSON."
        ),
        "prompt":  prompt,
        "stream":  False,
        "options": _CLASSIFIER_OPTIONS,
    }

    async with _ollama_semaphore:                            # ← SAME semaphore as summarizer + pipeline
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post(OLLAMA_URL, json=payload)
                    r.raise_for_status()
                    return r.json().get("response", "").strip()

            except httpx.ConnectError:
                logger.error("[CLASSIFIER] Ollama not running")
                return ""                                   # no retry — Ollama is off

            except httpx.TimeoutException:
                logger.error(f"[CLASSIFIER] Timed out after 30s (model: {_CLASSIFIER_MODEL})")
                return ""

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 404:
                    logger.error(
                        f"[CLASSIFIER] Model '{_CLASSIFIER_MODEL}' not found. "
                        f"Run: ollama pull {_CLASSIFIER_MODEL}"
                    )
                    return ""

                if status == 500 and attempt < 2:
                    wait = 2 * (attempt + 1)
                    logger.warning(f"[CLASSIFIER] HTTP 500, retry {attempt+1}/3 in {wait}s...")
                    await asyncio.sleep(wait)
                    continue

                logger.error(f"[CLASSIFIER] HTTP error: {e}")
                return ""

            except Exception as e:
                logger.error(f"[CLASSIFIER] Unexpected error: {e}")
                return ""

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API  (now async)
# ─────────────────────────────────────────────────────────────────────────────

async def classify(user_input: str) -> dict:
    """
    Async classify of user input into a source_type using a fast LLM call.

    Args:
        user_input: Any string — URL, pasted text, OCR output, headline, etc.

    Returns:
        dict with keys: source_type, confidence, reason, raw

    Never raises — always returns a safe fallback dict on any failure.
    """
    if not user_input or not user_input.strip():
        return _fallback("empty input")

    prompt = build_classifier_prompt(user_input)
    raw    = await _call_classifier(prompt)                  # ← await

    if not raw:
        return _fallback("LLM returned empty response")

    parsed = _extract_json(raw)

    if not parsed:
        return _fallback(f"could not parse JSON from: {raw[:80]}")

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
    """Returns False if source_type is one main.py cannot process."""
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
        ("https://youtube.com/watch?v=abc123",               "youtube_video"),
        ("https://github.com/openai/whisper",                "github_repo"),
        ("https://github.com/palinkiewicz",                  "github_user"),
        ("https://arxiv.org/abs/2303.08774",                 "arxiv_paper"),
        ("https://huggingface.co/mistralai/Mistral-7B-v0.1", "huggingface_model"),
        ("https://www.instagram.com/aditya.ksx/",            "instagram_profile"),
        ("https://linkedin.com/in/johndoe",                  "linkedin_profile"),
        ("https://adityaksx.vercel.app/",                    "unsupported"),
        ("https://medium.com/@user/my-article",              "medium_article"),
        ("https://reddit.com/r/Python/comments/abc",         "reddit_post"),
        ("def fibonacci(n): return n if n < 2 else ...",     "code_snippet"),
        ('{"name": "aditya", "role": "dev"}',                "json_data"),
        ("# My Notes\n- Point 1\n- Point 2\n**bold**",       "markdown"),
        ("India's Sarvam launches Indus AI chat app",         "news_headline"),
        ("FastAPI is a fast Python web framework.",           "plain_text"),
    ]

    async def main():
        print(f"\n{'Input':<52} {'Expected':<22} {'Got':<22} {'Conf':<8} {'✓/?'}")
        print("─" * 115)

        passed = failed = 0
        for inp, expected in tests:
            result = await classify(inp)                     # ← await
            got    = result["source_type"]
            conf   = result["confidence"]
            ok     = "✅" if got == expected else "❌"
            passed += 1 if got == expected else 0
            failed += 0 if got == expected else 1
            print(f"{inp[:51]:<52} {expected:<22} {got:<22} {conf:<8} {ok}")

        print(f"\n{passed}/{passed + failed} passed")

    asyncio.run(main())
