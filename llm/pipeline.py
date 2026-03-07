"""
llm/pipeline.py
---------------
Orchestrates the multi-stage LLM pipeline for the AI Resource Agent.

Stage 1 — classify()         Identify input type (delegates to llm_classifier.py)
Stage 2 — extract_guidance() Tell the processor what to focus on / skip
Stage 3 — enrich()           Fill gaps in cleaned data after processor runs
Stage 4 — summarize_data()   Final structured answer (lives in summarizer.py)

This file owns Stages 2 and 3 only.
Stage 1 is in llm/llm_classifier.py.
Stage 4 is in llm/summarizer.py.

Does NOT:
  - Build prompt strings      → llm/prompt_builder.py
  - Select models             → llm/summarizer.py (get_model)
  - Clean processor output    → utils/cleaner.py
  - Route input to processors → main.py

Public API consumed by main.py:
  classify(user_input)              → dict  (re-exported from llm_classifier)
  extract_guidance(input, type)     → dict  {focus_on, skip, infer, context}
  enrich(cleaned_data, guidance)    → dict  (cleaned_data + inferred fields)
"""

from __future__ import annotations

import asyncio
import json
import logging
import httpx

from llm.llm_classifier  import classify                        # Stage 1 — re-exported
from llm.prompt_builder  import build_guidance_prompt, build_enrich_prompt
from llm.summarizer      import call_llm, _ollama_semaphore     # ← share the SAME semaphore

logger = logging.getLogger(__name__)

try:
    from config import OLLAMA_URL  # type: ignore
except ImportError:
    OLLAMA_URL = "http://localhost:11434/api/generate"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIG FOR PIPELINE STAGES 2 & 3
# ─────────────────────────────────────────────────────────────────────────────

_PIPELINE_MODEL = "mistral:7b"

_CODE_TYPES: set[str] = {
    "github_repo", "github_file", "github_gist",
    "code_snippet", "notebook",
}

_PIPELINE_OPTIONS_FAST = {"temperature": 0.1, "num_predict": 350, "top_p": 0.9}
_PIPELINE_OPTIONS_CODE = {"temperature": 0.1, "num_predict": 350, "top_p": 0.9}


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _pipeline_model(source_type: str) -> str:
    if source_type in _CODE_TYPES:
        return "deepseek-coder:6.7b"
    return _PIPELINE_MODEL


def _pipeline_options(source_type: str) -> dict:
    if source_type in _CODE_TYPES:
        return _PIPELINE_OPTIONS_CODE
    return _PIPELINE_OPTIONS_FAST

# System message per task — passed in from extract_guidance/enrich
_PIPELINE_SYSTEM: dict[str, str] = {
    "guidance":   (
        "Return ONLY a valid JSON object. "
        "No explanation. No markdown. No extra text before or after the JSON."
    ),
    "enrichment": (
        "You are enriching content metadata. "
        "Return ONLY a valid JSON object with inferred fields. "
        "No explanation or markdown."
    ),
}

async def _call_pipeline(prompt: str, source_type: str = "", task: str = "guidance") -> str:
    """
    Async Ollama call for pipeline intermediate stages (2 & 3).
    Uses the SAME _ollama_semaphore from summarizer.py — ensures
    all Ollama calls across the entire app are serialised.

    Separate from summarizer.call_llm() because:
      - Uses different models (mistral:7b / deepseek-coder)
      - Uses lower num_predict (only needs JSON, not prose)
      - Has a shorter timeout (30s — must be fast)
    """
    model   = _pipeline_model(source_type)
    options = _pipeline_options(source_type)

    payload = {
        "model":   model,
        "system":  _PIPELINE_SYSTEM.get(task, _PIPELINE_SYSTEM["guidance"]),
        "prompt":  prompt,
        "stream":  False,
        "options": options,
    }


    async with _ollama_semaphore:                        # ← SAME semaphore as summarizer.py
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post(OLLAMA_URL, json=payload)
                    r.raise_for_status()
                    return r.json().get("response", "").strip()

            except httpx.ConnectError:
                logger.error("[PIPELINE] Ollama not running")
                return ""                               # no retry — Ollama is off

            except httpx.TimeoutException:
                logger.warning(f"[PIPELINE] Timed out (model: {model})")
                return ""

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 404:
                    logger.error(
                        f"[PIPELINE] Model '{model}' not found. "
                        f"Run: ollama pull {model}"
                    )
                    return ""

                if status == 500 and attempt < 2:
                    wait = 2 * (attempt + 1)
                    logger.warning(f"[PIPELINE] HTTP 500, retry {attempt+1}/3 in {wait}s...")
                    await asyncio.sleep(wait)
                    continue

                logger.error(f"[PIPELINE] HTTP error: {e}")
                return ""

            except Exception as e:
                logger.error(f"[PIPELINE] Unexpected error: {e}")
                return ""

    return ""


def _extract_json(text: str) -> dict:
    """
    Safely extract a JSON object from an LLM response.
    Handles:
      - Markdown fences
      - Surrounding explanation text
      - Trailing commas
      - Literal newlines/control characters inside string values
      - Multiple JSON objects — takes the FIRST complete one   ← NEW
    """
    if not text:
        return {}

    import re as _re

    text = text.strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(l for l in lines if not l.strip().startswith("```"))

    # Find first opening brace
    start = text.find("{")
    if start == -1:
        logger.warning(f"[PIPELINE] No JSON found in: {text[:100]}")
        return {}

    # ── NEW: walk braces to find the FIRST complete matching } ───────────
    depth = 0
    end   = -1
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
        logger.warning(f"[PIPELINE] Unmatched braces in: {text[:100]}")
        return {}

    raw_json = text[start:end]
    # ─────────────────────────────────────────────────────────────────────

    # Sanitize control characters inside quoted strings
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
        fixed = _re.sub(r",\s*([}\]])", r"\1", raw_json)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            logger.warning(f"[PIPELINE] JSON parse failed: {e} | raw: {raw_json[:100]}")
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — CLASSIFY  (re-exported from llm_classifier.py)
# ─────────────────────────────────────────────────────────────────────────────

__all__ = ["classify", "extract_guidance", "enrich"]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — EXTRACTION GUIDANCE  (now async)
# ─────────────────────────────────────────────────────────────────────────────

async def extract_guidance(user_input: str, source_type: str) -> dict:
    """
    Stage 2: Ask the LLM what the processor should focus on and skip.
    Now async — awaits _call_pipeline() without blocking the event loop.

    Returns dict with keys: focus_on, skip, infer, context
    Never raises — returns safe empty defaults on any failure.
    """
    if not user_input or not source_type:
        return _default_guidance()

    prompt = build_guidance_prompt(user_input, source_type)
    raw    = await _call_pipeline(prompt, source_type,task="guidance")          # ← await

    if not raw:
        logger.warning(f"[S2-GUIDANCE] Empty response for source_type={source_type}")
        return _default_guidance()

    parsed = _extract_json(raw)

    if not parsed:
        logger.warning(f"[S2-GUIDANCE] No JSON parsed for source_type={source_type}")
        return _default_guidance()

    result = {
        "focus_on": parsed.get("focus_on", []),
        "skip":     parsed.get("skip",     []),
        "infer":    parsed.get("infer",    []),
        "context":  parsed.get("context",  ""),
    }

    for key in ("focus_on", "skip", "infer"):
        if isinstance(result[key], str):
            result[key] = [result[key]]
        elif not isinstance(result[key], list):
            result[key] = []

    if not isinstance(result["context"], str):
        result["context"] = str(result["context"])

    logger.info(
        f"[S2-GUIDANCE] focus={result['focus_on'][:2]}  "
        f"skip={result['skip'][:2]}  "
        f"infer={result['infer'][:2]}"
    )
    print(
        f"  [S2-EXTRACT]  focus={result['focus_on'][:2]}  "
        f"skip={result['skip'][:2]}"
    )

    return result


def _default_guidance() -> dict:
    return {"focus_on": [], "skip": [], "infer": [], "context": ""}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — ENRICHMENT  (now async)
# ─────────────────────────────────────────────────────────────────────────────

async def enrich(cleaned_data: dict, guidance: dict) -> dict:
    """
    Stage 3: Fill gaps in cleaned processor output using LLM inference.
    Now async — awaits _call_pipeline() without blocking the event loop.

    Never removes or overwrites existing fields — only ADDS new ones.
    Never raises — returns cleaned_data unchanged on any failure.
    """
    if not cleaned_data:
        return cleaned_data

    source_type = cleaned_data.get("source_type", "")

    infer_list  = guidance.get("infer", [])
    has_content = any(
        isinstance(v, str) and len(v) > 30
        for k, v in cleaned_data.items()
        if k not in ("source_type", "url", "title")
    )

    if not infer_list and not has_content:
        logger.info("[S3-ENRICH] Nothing to infer, skipping enrichment")
        return cleaned_data

    prompt   = build_enrich_prompt(cleaned_data, guidance)
    raw      = await _call_pipeline(prompt, source_type,task="enrichment")        # ← await

    if not raw:
        logger.warning("[S3-ENRICH] Empty response, returning cleaned data unchanged")
        return cleaned_data

    inferred = _extract_json(raw)

    if not inferred:
        logger.info("[S3-ENRICH] Nothing new to add")
        print("  [S3-ENRICH]   nothing new to add")
        return cleaned_data

    _ALLOWED_INFERRED_FIELDS: set[str] = {
        "inferred_audience",
        "inferred_difficulty",
        "related_tools",
        "missing_context",
        "inferred_use_case",      # what someone would use this for
        "inferred_prerequisites", # what you need to know first
        "inferred_category",      # e.g. "DevOps tool", "ML paper", "tutorial"
        "key_entities",           # people, orgs, products mentioned
    }

    added = []
    for key, value in inferred.items():
        if (
            key in _ALLOWED_INFERRED_FIELDS
            and key not in cleaned_data
            and isinstance(value, str)
            and value.strip()
        ):
            cleaned_data[key] = value.strip()
            added.append(key)

    if added:
        logger.info(f"[S3-ENRICH] Added fields: {added}")
        print(f"  [S3-ENRICH]   added fields: {added}")
    else:
        logger.info("[S3-ENRICH] No valid new fields to add")
        print("  [S3-ENRICH]   no valid new fields to add")

    return cleaned_data


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE HEALTH CHECK  (now async)
# ─────────────────────────────────────────────────────────────────────────────

async def check_pipeline_models() -> dict[str, bool]:
    """
    Async check which pipeline models are available in Ollama.
    Returns dict of model_name → is_available.
    """
    required_models = {_PIPELINE_MODEL, "deepseek-coder:6.7b"}
    status: dict[str, bool] = {}

    try:
        base_url = OLLAMA_URL.replace("/api/generate", "")
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{base_url}/api/tags")
            if r.status_code == 200:
                available = {m["name"] for m in r.json().get("models", [])}
                for model in required_models:
                    status[model] = model in available
            else:
                for model in required_models:
                    status[model] = False
    except Exception:
        for model in required_models:
            status[model] = False

    return status


# ─────────────────────────────────────────────────────────────────────────────
# CLI TEST  (python -m llm.pipeline)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)

    async def main():
        print("=" * 60)
        print("Pipeline Model Health Check")
        print("=" * 60)
        model_status = await check_pipeline_models()            # ← await
        for model, available in model_status.items():
            icon = "✅" if available else "❌"
            tip  = "" if available else f"  ← run: ollama pull {model}"
            print(f"  {icon} {model}{tip}")

        print()
        print("=" * 60)
        print("Stage 1 — Classify")
        print("=" * 60)
        test_inputs = [
            "https://github.com/tiangolo/fastapi",
            "https://arxiv.org/abs/2303.08774",
            "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
            "India's Sarvam launches Indus AI chat app",
        ]

        for inp in test_inputs:
            result = await classify(inp)                        # ← await (once llm_classifier is fixed)
            print(f"  Input : {inp[:55]}")
            print(f"  Result: {result['source_type']} ({result['confidence']}) — {result['reason']}")
            print()

        print("=" * 60)
        print("Stage 2 — Extraction Guidance")
        print("=" * 60)
        guidance = await extract_guidance(                      # ← await
            "https://github.com/tiangolo/fastapi",
            "github_repo"
        )
        print(f"  focus_on : {guidance['focus_on']}")
        print(f"  skip     : {guidance['skip']}")
        print(f"  infer    : {guidance['infer']}")
        print(f"  context  : {guidance['context']}")

        print()
        print("=" * 60)
        print("Stage 3 — Enrichment")
        print("=" * 60)
        mock_cleaned = {
            "source_type": "github_repo",
            "title":       "FastAPI",
            "description": "FastAPI framework, high performance, easy to learn.",
            "readme":      "FastAPI is a modern, fast web framework for building APIs with Python 3.7+.",
        }
        mock_guidance = {
            "focus_on": ["README", "tech stack"],
            "skip":     ["CI config"],
            "infer":    ["target audience", "related tools", "difficulty level"],
            "context":  "This is an open source Python web framework on GitHub.",
        }
        enriched   = await enrich(mock_cleaned, mock_guidance)  # ← await
        new_fields = {k: v for k, v in enriched.items() if k not in mock_cleaned}
        print(f"  New fields added: {list(new_fields.keys())}")
        for k, v in new_fields.items():
            print(f"    {k}: {v}")

    asyncio.run(main())
