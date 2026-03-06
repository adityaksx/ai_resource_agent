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

import json
import logging
import requests

from llm.llm_classifier  import classify                       # Stage 1 — re-exported
from llm.prompt_builder  import build_guidance_prompt, build_enrich_prompt
from llm.summarizer      import call_llm                       # Stage 4 raw caller

logger = logging.getLogger(__name__)

# Try to read Ollama URL from config; fall back to default
try:
    from config import OLLAMA_URL  # type: ignore
except ImportError:
    OLLAMA_URL = "http://localhost:11434/api/generate"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIG FOR PIPELINE STAGES 2 & 3
# ─────────────────────────────────────────────────────────────────────────────

# Stage 2 + 3 use a fast model — they just need JSON decisions, not rich prose
_PIPELINE_MODEL = "mistral:7b"

# Code-related source types get deepseek-coder for guidance/enrichment too
_CODE_TYPES: set[str] = {
    "github_repo", "github_file", "github_gist",
    "code_snippet", "notebook",
}

_PIPELINE_OPTIONS_FAST = {
    "temperature": 0.1,    # slight creativity for inference, but mostly deterministic
    "num_predict": 350,
    "top_p":       0.9,
}

_PIPELINE_OPTIONS_CODE = {
    "temperature": 0.1,
    "num_predict": 350,
    "top_p":       0.9,
}


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _pipeline_model(source_type: str) -> str:
    """Select model for pipeline intermediate stages (2 & 3)."""
    if source_type in _CODE_TYPES:
        return "deepseek-coder:6.7b"
    return _PIPELINE_MODEL


def _pipeline_options(source_type: str) -> dict:
    """Select generation options for pipeline intermediate stages."""
    if source_type in _CODE_TYPES:
        return _PIPELINE_OPTIONS_CODE
    return _PIPELINE_OPTIONS_FAST


def _call_pipeline(prompt: str, source_type: str = "") -> str:
    """
    Raw Ollama call for pipeline intermediate stages (2 & 3).
    Separate from summarizer.call_llm() because:
      - Uses different models (mistral:7b / deepseek-coder)
      - Uses lower num_predict (only needs JSON, not prose)
      - Has a shorter timeout (30s — must be fast)
    """
    model   = _pipeline_model(source_type)
    options = _pipeline_options(source_type)

    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model":   model,
                "prompt":  prompt,
                "stream":  False,
                "options": options,
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()

    except requests.exceptions.ConnectionError:
        logger.error("[PIPELINE] Ollama not running")
        return ""
    except requests.exceptions.Timeout:
        logger.warning(f"[PIPELINE] Timed out (model: {model})")
        return ""
    except requests.exceptions.HTTPError as e:
        if "404" in str(e):
            logger.error(
                f"[PIPELINE] Model '{model}' not found. "
                f"Run: ollama pull {model}"
            )
        else:
            logger.error(f"[PIPELINE] HTTP error: {e}")
        return ""
    except Exception as e:
        logger.error(f"[PIPELINE] Unexpected error: {e}")
        return ""


def _extract_json(text: str) -> dict:
    """
    Safely extract a JSON object from an LLM response.
    Handles markdown fences, surrounding explanation text, trailing commas.
    """
    if not text:
        return {}

    # Strip markdown code fences
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(
            l for l in lines if not l.strip().startswith("```")
        )

    start = text.find("{")
    end   = text.rfind("}") + 1

    if start == -1 or end <= start:
        logger.warning(f"[PIPELINE] No JSON found in: {text[:100]}")
        return {}

    raw_json = text[start:end]

    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        # Attempt to fix trailing commas (common LLM mistake)
        import re
        fixed = re.sub(r",\s*([}\]])", r"\1", raw_json)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            logger.warning(f"[PIPELINE] JSON parse failed: {e} | raw: {raw_json[:100]}")
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — CLASSIFY  (re-exported from llm_classifier.py)
# ─────────────────────────────────────────────────────────────────────────────

# classify() is imported directly from llm_classifier and re-exported here
# so main.py only needs: from llm.pipeline import classify, extract_guidance, enrich
# No logic duplication — pure re-export.
__all__ = ["classify", "extract_guidance", "enrich"]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — EXTRACTION GUIDANCE
# ─────────────────────────────────────────────────────────────────────────────

def extract_guidance(user_input: str, source_type: str) -> dict:
    """
    Stage 2: Ask the LLM what the processor should focus on and skip.

    This runs AFTER classify() and BEFORE the processor output is cleaned.
    The guidance dict is passed to enrich() in Stage 3 to help fill gaps.

    Args:
        user_input:  The original URL or text (first 600 chars used in prompt).
        source_type: Detected type from Stage 1 (e.g. "github_repo").

    Returns:
        dict with keys:
          focus_on : list[str]  — fields or sections to prioritise
          skip     : list[str]  — noise to ignore
          infer    : list[str]  — things that can be inferred later
          context  : str        — one-sentence background for this content type

    Never raises — returns safe empty defaults on any failure.

    Example:
        guidance = extract_guidance("https://github.com/tiangolo/fastapi", "github_repo")
        # → {
        #     "focus_on": ["README", "tech stack", "installation"],
        #     "skip":     ["test files", "CI config"],
        #     "infer":    ["target audience", "maturity level"],
        #     "context":  "This is an open source Python web framework."
        #   }
    """
    if not user_input or not source_type:
        return _default_guidance()

    prompt = build_guidance_prompt(user_input, source_type)
    raw    = _call_pipeline(prompt, source_type)

    if not raw:
        logger.warning(f"[S2-GUIDANCE] Empty response for source_type={source_type}")
        return _default_guidance()

    parsed = _extract_json(raw)

    if not parsed:
        logger.warning(f"[S2-GUIDANCE] No JSON parsed for source_type={source_type}")
        return _default_guidance()

    # Sanitize — ensure correct types for each key
    result = {
        "focus_on": parsed.get("focus_on", []),
        "skip":     parsed.get("skip",     []),
        "infer":    parsed.get("infer",    []),
        "context":  parsed.get("context",  ""),
    }

    # Ensure lists are actually lists (model sometimes returns strings)
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
    """Safe fallback when Stage 2 fails — pipeline continues unaffected."""
    return {
        "focus_on": [],
        "skip":     [],
        "infer":    [],
        "context":  "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def enrich(cleaned_data: dict, guidance: dict) -> dict:
    """
    Stage 3: Fill gaps in cleaned processor output using LLM inference.

    Runs AFTER utils/cleaner.py has cleaned the processor dict,
    and BEFORE summarizer.summarize_data() builds the final answer.

    The LLM looks at what fields are already present and what the
    Stage 2 guidance says should be inferred, then adds new fields
    (inferred_audience, inferred_difficulty, related_tools, missing_context).

    Args:
        cleaned_data: Dict from clean_processor_output() — already cleaned.
        guidance:     Dict from extract_guidance() — Stage 2 output.

    Returns:
        The same cleaned_data dict, augmented with any inferred fields.
        Never removes or overwrites existing fields — only ADDS new ones.

    Never raises — returns cleaned_data unchanged on any failure.

    Example:
        enriched = enrich(cleaned, guidance)
        # cleaned_data now has extra keys like:
        # "inferred_audience":   "backend Python developers"
        # "inferred_difficulty": "Intermediate"
        # "related_tools":       "Flask, Django, Starlette"
        # "missing_context":     "This project has 70k+ GitHub stars"
    """
    if not cleaned_data:
        return cleaned_data

    source_type = cleaned_data.get("source_type", "")

    # Skip enrichment if guidance has nothing to infer
    # and we have no meaningful content to reason about
    infer_list = guidance.get("infer", [])
    has_content = any(
        isinstance(v, str) and len(v) > 30
        for k, v in cleaned_data.items()
        if k not in ("source_type", "url", "title")
    )

    if not infer_list and not has_content:
        logger.info("[S3-ENRICH] Nothing to infer, skipping enrichment")
        return cleaned_data

    prompt   = build_enrich_prompt(cleaned_data, guidance)
    raw      = _call_pipeline(prompt, source_type)

    if not raw:
        logger.warning("[S3-ENRICH] Empty response, returning cleaned data unchanged")
        return cleaned_data

    inferred = _extract_json(raw)

    if not inferred:
        logger.info("[S3-ENRICH] Nothing new to add")
        print("  [S3-ENRICH]   nothing new to add")
        return cleaned_data

    # ── Validate inferred fields ──────────────────────────────────────────
    _ALLOWED_INFERRED_FIELDS: set[str] = {
        "inferred_audience",
        "inferred_difficulty",
        "related_tools",
        "missing_context",
    }

    added = []
    for key, value in inferred.items():
        # Only add fields that are:
        #  1. In the allowed set (prevent LLM from hallucinating random keys)
        #  2. Not already present in cleaned_data (never overwrite real data)
        #  3. Non-empty strings
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
# PIPELINE HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_pipeline_models() -> dict[str, bool]:
    """
    Check which pipeline models are available in Ollama.
    Returns dict of model_name → is_available.

    Usage:
        status = check_pipeline_models()
        # → {"mistral:7b": True, "deepseek-coder:6.7b": False}
    """
    required_models = {
        _PIPELINE_MODEL,
        "deepseek-coder:6.7b",
    }

    status: dict[str, bool] = {}

    try:
        base_url = OLLAMA_URL.replace("/api/generate", "")
        r = requests.get(f"{base_url}/api/tags", timeout=5)
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
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Pipeline Model Health Check")
    print("=" * 60)
    model_status = check_pipeline_models()
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
        "India's Sarvam launches Indus AI chat app as competition heats up",
    ]

    for inp in test_inputs:
        result = classify(inp)
        print(f"  Input : {inp[:55]}")
        print(f"  Result: {result['source_type']} ({result['confidence']}) — {result['reason']}")
        print()

    print("=" * 60)
    print("Stage 2 — Extraction Guidance")
    print("=" * 60)
    guidance = extract_guidance(
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
        "description": "FastAPI framework, high performance, easy to learn, fast to code, ready for production.",
        "readme":      "FastAPI is a modern, fast web framework for building APIs with Python 3.7+.",
    }
    mock_guidance = {
        "focus_on": ["README", "tech stack"],
        "skip":     ["CI config"],
        "infer":    ["target audience", "related tools", "difficulty level"],
        "context":  "This is an open source Python web framework on GitHub.",
    }
    enriched = enrich(mock_cleaned, mock_guidance)
    new_fields = {
        k: v for k, v in enriched.items()
        if k not in mock_cleaned
    }
    print(f"  New fields added: {list(new_fields.keys())}")
    for k, v in new_fields.items():
        print(f"    {k}: {v}")
