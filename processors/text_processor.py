"""
processors/text_processor.py
-----------------------------
Processes raw text input passed directly by the user — not a URL.

Handles all raw text sub-types:
  plain_text    → user notes, pasted articles, meeting notes
  code_snippet  → any programming language, detected automatically
  markdown      → README-style or documentation text
  json_data     → structured JSON blobs
  news_headline → short news snippets

Returns a structured dict ready for:
  utils/cleaner.py          → clean_processor_output()
  llm/summarizer.py         → summarize_data()

Does NOT:
  - Fetch content from URLs  → web_processor.py / youtube_processor.py etc.
  - Call the LLM             → llm/summarizer.py
  - Clean the text           → utils/cleaner.py
  - Build prompts            → llm/prompt_builder.py

Called by:
  main.py → process_text_input()
  main.py → process_local_file() (reads file → passes text here)
"""

from __future__ import annotations

import json
import re
from utils.source_detector import detect_source   # public API — not _detect_raw_text


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# Heuristic patterns to detect programming language from a code snippet.
# First match wins — ordered from most distinctive to least.
_LANGUAGE_PATTERNS: list[tuple[str, str]] = [
    # Python
    (r"^\s*(def |class |import |from |async def |@\w+\n)", "Python"),
    (r"if __name__\s*==\s*['\"]__main__['\"]",             "Python"),
    # JavaScript / TypeScript
    (r"^\s*(const |let |var |function |=>|export |import )", "JavaScript"),
    (r":\s*(string|number|boolean|any)\b",                  "TypeScript"),
    # Java / Kotlin
    (r"^\s*(public|private|protected)\s+(static\s+)?\w+\s+\w+\(", "Java"),
    (r"^\s*(fun |val |var )\w+",                            "Kotlin"),
    # C / C++
    (r"^\s*#include\s*<",                                   "C/C++"),
    (r"^\s*int\s+main\s*\(",                                "C/C++"),
    # C#
    (r"^\s*(using\s+System|namespace\s+\w+)",               "C#"),
    # Go
    (r"^\s*(package\s+main|func\s+\w+\()",                  "Go"),
    # Rust
    (r"^\s*(fn\s+\w+|let\s+mut\s+)",                        "Rust"),
    # Ruby
    (r"^\s*(def\s+\w+|end$|require\s+['\"])",               "Ruby"),
    # PHP
    (r"<\?php",                                             "PHP"),
    # Swift
    (r"^\s*(var\s+\w+:\s*\w+|func\s+\w+\(.*\)\s*->)",      "Swift"),
    # Shell
    (r"^#!/(bin/bash|bin/sh|usr/bin/env\s)",                "Shell"),
    (r"^\s*(echo |cd |ls |mkdir |chmod |sudo |export )",    "Shell"),
    # SQL
    (r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\s", "SQL"),
    # HTML
    (r"<!DOCTYPE\s+html|<html",                             "HTML"),
    # CSS
    (r"^\s*[\w.#:\[\]]+\s*\{[\s\S]*?\}",                   "CSS"),
    # YAML
    (r"^---\s*$|^\w+:\s*$",                                 "YAML"),
    # JSON (already handled separately, but fallback)
    (r"^\s*[\{\[]",                                         "JSON"),
]


def _detect_language(code: str) -> str:
    """
    Detect the programming language of a code snippet.
    Returns a human-readable language name, or 'Unknown' if no pattern matches.
    """
    for pattern, language in _LANGUAGE_PATTERNS:
        if re.search(pattern, code, re.MULTILINE | re.IGNORECASE):
            return language
    return "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# TITLE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_title(text: str, source_type: str) -> str:
    """
    Extract a meaningful title from raw text.

    Strategy by type:
      markdown     → first # heading line
      json_data    → "JSON: <top-level keys>"
      code_snippet → "Code: <first function or class name>"
      plain_text   → first non-empty line, max 80 chars, cut at word boundary
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return "Untitled"

    if source_type == "markdown":
        for line in lines:
            if line.startswith("#"):
                return line.lstrip("#").strip()[:80]

    if source_type == "json_data":
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                keys = list(parsed.keys())[:4]
                return f"JSON: {', '.join(keys)}"
        except Exception:
            pass
        return "JSON Data"

    if source_type == "code_snippet":
        # Try to find first function or class definition
        match = re.search(
            r"^\s*(def |class |function |func |fn |sub )\s*(\w+)",
            text, re.MULTILINE | re.IGNORECASE
        )
        if match:
            return f"Code: {match.group(2)}"

    # Default: first line, cut at word boundary ≤ 80 chars
    first_line = lines[0]
    if len(first_line) <= 80:
        return first_line
    cut = first_line[:80].rsplit(" ", 1)[0]
    return cut or first_line[:80]


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSING
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_content(text: str) -> dict:
    """
    Extract a structured description of a JSON blob.
    Returns a dict with description + raw content for the LLM.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"content": text}   # malformed JSON — treat as plain text

    if isinstance(parsed, dict):
        keys        = list(parsed.keys())
        description = f"JSON object with {len(keys)} keys: {', '.join(str(k) for k in keys[:10])}"
        nested      = [k for k, v in parsed.items() if isinstance(v, (dict, list))]
        if nested:
            description += f". Nested structures in: {', '.join(nested[:5])}"
    elif isinstance(parsed, list):
        description = f"JSON array with {len(parsed)} items"
        if parsed and isinstance(parsed[0], dict):
            sample_keys = list(parsed[0].keys())[:5]
            description += f". Each item has keys: {', '.join(sample_keys)}"
    else:
        description = f"JSON value: {str(parsed)[:100]}"

    return {
        "description": description,
        "content":     text,          # raw JSON string for the LLM to read
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

def process_text(text: str, source_type: str = "") -> dict:
    """
    Process raw user-provided text and return a structured dict.

    Args:
        text:        Raw string input from the user.
        source_type: Optional — pre-classified type from Stage 1 (classify()).
                     If empty, detected locally via detect_source().
                     main.py always passes this after running classify().

    Returns:
        dict ready for clean_processor_output() then summarize_data().
        Keys always present:
          source_type  : str
          content      : str  (the full original text)
          title        : str  (extracted meaningful title)
          char_count   : int
          word_count   : int
          line_count   : int

        Keys added by type:
          language     : str  (code_snippet only)
          description  : str  (json_data only)

    Never raises — returns {"error": ...} on bad input.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return {"source_type": "plain_text", "error": "Empty text input"}

    text = text.strip()

    # ── Source type: use pre-classified if provided, else detect locally ──
    # main.py calls classify() first (Stage 1), then passes source_type here.
    # For direct calls (e.g. process_local_file), detect_source() handles it.
    if not source_type:
        source_type = detect_source(text)

    # Normalise: detect_source on a URL would return "web" — keep as plain_text
    _TEXT_TYPES = {
        "plain_text", "code_snippet", "markdown",
        "json_data", "news_headline", "unknown",
    }
    if source_type not in _TEXT_TYPES:
        source_type = "plain_text"

    # ── Base dict ─────────────────────────────────────────────────────────
    result: dict = {
        "source_type": source_type,
        "content":     text,
        "title":       _extract_title(text, source_type),
        "char_count":  len(text),
        "word_count":  len(text.split()),
        "line_count":  len(text.splitlines()),
    }

    # ── Type-specific enrichment ──────────────────────────────────────────

    if source_type == "code_snippet":
        result["language"] = _detect_language(text)

    elif source_type == "json_data":
        json_fields = _parse_json_content(text)
        result.update(json_fields)    # adds description + keeps content

    elif source_type == "markdown":
        # Count structural elements for context
        headings = len(re.findall(r"^#{1,6}\s", text, re.MULTILINE))
        bullets  = len(re.findall(r"^\s*[-*]\s", text, re.MULTILINE))
        if headings:
            result["description"] = (
                f"Markdown document with {headings} heading(s)"
                + (f" and {bullets} bullet point(s)" if bullets else "")
            )

    elif source_type == "news_headline":
        # Short — just a headline or snippet; no extra processing needed
        # Title is already set from first line; content is the full text
        pass

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI TEST  (python -m processors.text_processor)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint

    tests = [
        # Plain text
        ("Just some thoughts about AI and how it's changing software development.", ""),
        # Code snippet — no pre-classification
        ("def greet(name: str) -> str:\n    return f'Hello, {name}!'", ""),
        # Code snippet — pre-classified by Stage 1
        ("const fetch = require('node-fetch')\nfetch('https://api.example.com')", "code_snippet"),
        # Markdown
        ("# My Notes\n\n## Section 1\n- Point A\n- Point B\n\n**Bold** and `code`", ""),
        # JSON
        ('{"name": "aditya", "role": "developer", "skills": ["Python", "FastAPI"]}', ""),
        # News headline
        ("India's Sarvam launches Indus AI chatbot to compete with ChatGPT", "news_headline"),
        # Empty input
        ("", ""),
    ]

    for text, pre_type in tests:
        print(f"\n{'─' * 60}")
        print(f"Input     : {text[:55]!r}")
        print(f"Pre-type  : {pre_type!r}")
        result = process_text(text, source_type=pre_type)
        pprint.pprint(result, width=80, sort_dicts=False)
