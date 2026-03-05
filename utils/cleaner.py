"""
cleaner.py
----------
Cleans and normalises raw text extracted from any source before
it is passed to the LLM.

Handles:
  - Plain text / web articles
  - Transcripts (YouTube, Whisper, VTT)
  - Code files / notebooks
  - OCR output
  - Social media captions / comments
  - Markdown / RST documents
  - JSON / structured data fields
  - PDF / document text

Key improvements over v1:
  - Preserves code blocks instead of stripping them
  - Mode-aware cleaning (prose vs code vs transcript vs ocr)
  - Configurable via CleanConfig dataclass
  - Token-budget trimming (for LLM context limits)
  - Near-duplicate detection (fuzzy, not just exact)
  - Structured CleanResult dataclass
  - Keeps URLs when needed (mode-aware)
  - Handles Unicode, smart quotes, zero-width chars
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

@dataclass
class CleanConfig:
    # Cleaning toggles
    remove_urls:        bool  = True
    remove_hashtags:    bool  = True
    remove_emojis:      bool  = False   # keep by default — useful context
    remove_html_tags:   bool  = True
    fix_unicode:        bool  = True
    normalize_spaces:   bool  = True
    normalize_quotes:   bool  = True

    # Deduplication
    dedupe_lines:       bool  = True
    dedupe_threshold:   float = 0.85    # Jaccard similarity threshold

    # Compression
    min_sentence_len:   int   = 15      # chars — shorter sentences dropped
    max_sentences:      int   = 100
    max_comments:       int   = 30
    min_comment_len:    int   = 10

    # Token budget (rough: 1 token ≈ 4 chars)
    max_tokens:         Optional[int] = None   # e.g. 3000 for GPT-3.5

    # Mode — controls what gets preserved
    # "prose" | "transcript" | "code" | "ocr" | "social"
    mode:               str   = "prose"


# Default configs per source type
CONFIGS = {
    "prose":      CleanConfig(mode="prose"),
    "transcript": CleanConfig(mode="transcript", remove_urls=True,
                              remove_hashtags=False, max_sentences=200),
    "code":       CleanConfig(mode="code",       remove_urls=False,
                              remove_hashtags=False, remove_emojis=False,
                              dedupe_lines=False),
    "ocr":        CleanConfig(mode="ocr",        remove_urls=True,
                              min_sentence_len=5, dedupe_threshold=0.9),
    "social":     CleanConfig(mode="social",     remove_urls=True,
                              remove_hashtags=True, remove_emojis=True,
                              max_sentences=50),
}


# ─────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────

@dataclass
class CleanResult:
    text:           str
    mode:           str
    original_chars: int
    cleaned_chars:  int
    sentences:      int
    duplicates_removed: int = 0
    truncated:      bool    = False

    @property
    def compression_ratio(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return round(1 - self.cleaned_chars / self.original_chars, 3)

    def __str__(self):
        return self.text


# ─────────────────────────────────────────────
# LOW-LEVEL CLEANERS
# ─────────────────────────────────────────────

def _fix_unicode(text: str) -> str:
    """Normalize unicode, remove zero-width / control characters."""
    text = unicodedata.normalize("NFKC", text)
    # Remove zero-width spaces, BOM, soft hyphens, etc.
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)
    # Remove other control characters (keep \n \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def _normalize_quotes(text: str) -> str:
    """Replace smart quotes with straight quotes."""
    replacements = {
        "\u2018": "'", "\u2019": "'",   # ' '
        "\u201c": '"', "\u201d": '"',   # " "
        "\u2013": "-", "\u2014": "--",  # – —
        "\u2026": "...",                # …
    }
    for smart, straight in replacements.items():
        text = text.replace(smart, straight)
    return text


def _remove_html(text: str) -> str:
    """Strip HTML/XML tags and decode common entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    entities = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&apos;": "'", "&nbsp;": " ",
        "&#39;": "'",
    }
    for ent, char in entities.items():
        text = text.replace(ent, char)
    return text


def _remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def _remove_hashtags(text: str) -> str:
    return re.sub(r"#\w+", "", text)


def _remove_emojis(text: str) -> str:
    """Remove emoji and most symbols while keeping punctuation."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"   # emoticons
        "\U0001F300-\U0001F5FF"   # symbols & pictographs
        "\U0001F680-\U0001F6FF"   # transport & map
        "\U0001F1E0-\U0001F1FF"   # flags
        "\U00002700-\U000027BF"   # dingbats
        "\U0001F900-\U0001F9FF"   # supplemental symbols
        "\U00002600-\U000026FF"   # misc symbols
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub("", text)


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs to one; preserve paragraph breaks."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ newlines to 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse inline spaces/tabs
    text = re.sub(r"[^\S\n]+", " ", text)
    # Clean up spaces around newlines
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _remove_vtt_artifacts(text: str) -> str:
    """Remove VTT/SRT timestamp and header lines from raw transcripts."""
    text = re.sub(r"\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}", "", text)
    text = re.sub(r"^\d+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^WEBVTT.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", "", text)    # VTT inline tags <c>…</c>
    return text


def _remove_boilerplate(text: str) -> str:
    """Remove common web boilerplate phrases."""
    patterns = [
        r"cookie[s]? policy",
        r"accept all cookies",
        r"privacy policy",
        r"terms of (service|use)",
        r"all rights reserved",
        r"subscribe (to our newsletter|now)",
        r"click here to (read|view|see)",
        r"advertisement",
        r"skip to (main )?content",
        r"share this (article|post|story)",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text


# ─────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def deduplicate(
    lines: list[str],
    threshold: float = 0.85,
) -> tuple[list[str], int]:
    """
    Remove near-duplicate lines using Jaccard similarity.
    Returns (deduped_lines, num_removed).
    """
    seen:   list[str] = []
    result: list[str] = []
    removed = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append(line)
            continue

        is_dup = any(_jaccard(stripped, s) >= threshold for s in seen)
        if is_dup:
            removed += 1
        else:
            seen.append(stripped)
            result.append(line)

    return result, removed


# ─────────────────────────────────────────────
# SENTENCE SPLITTING
# ─────────────────────────────────────────────

def split_sentences(text: str, min_len: int = 15) -> list[str]:
    """
    Split text into sentences. Handles common abbreviations
    to avoid false splits (Mr., Dr., U.S., etc.).
    """
    # Protect common abbreviations
    abbrevs = r"(?<!\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx|est|vol|p|pp|fig|no)\b)"
    pattern = abbrevs + r"(?<=[.!?])\s+"
    parts = re.split(pattern, text)
    return [s.strip() for s in parts if len(s.strip()) >= min_len]


# ─────────────────────────────────────────────
# TOKEN BUDGET
# ─────────────────────────────────────────────

def trim_to_token_budget(text: str, max_tokens: int) -> tuple[str, bool]:
    """
    Trim text to approximately max_tokens (1 token ≈ 4 chars).
    Trims at sentence boundary where possible.
    Returns (trimmed_text, was_truncated).
    """
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text, False

    # Try to cut at last sentence boundary within budget
    chunk = text[:max_chars]
    last_break = max(chunk.rfind(". "), chunk.rfind(".\n"))
    if last_break > max_chars * 0.7:
        chunk = chunk[:last_break + 1]

    return chunk.strip(), True


# ─────────────────────────────────────────────
# MAIN CLEAN FUNCTION
# ─────────────────────────────────────────────

def clean_text(
    text: str,
    config: CleanConfig | None = None,
) -> CleanResult:
    """
    Full cleaning pipeline. Returns a CleanResult.
    Pass a CleanConfig or use one of the presets via clean(text, mode=...).
    """
    if not text or not isinstance(text, str):
        return CleanResult(text="", mode="prose",
                           original_chars=0, cleaned_chars=0, sentences=0)

    cfg = config or CleanConfig()
    original_chars = len(text)

    # ── Step 1: Unicode & encoding fixes ──────
    if cfg.fix_unicode:
        text = _fix_unicode(text)
    if cfg.normalize_quotes:
        text = _normalize_quotes(text)

    # ── Step 2: Format-specific pre-cleaning ──
    if cfg.mode == "transcript":
        text = _remove_vtt_artifacts(text)
    if cfg.remove_html_tags:
        text = _remove_html(text)

    # ── Step 3: Code mode — skip destructive cleaning ──
    if cfg.mode == "code":
        if cfg.normalize_spaces:
            text = _normalize_whitespace(text)
        sentences = text.splitlines()
        return CleanResult(
            text=text,
            mode=cfg.mode,
            original_chars=original_chars,
            cleaned_chars=len(text),
            sentences=len(sentences),
        )

    # ── Step 4: Content removal ───────────────
    if cfg.remove_urls:
        text = _remove_urls(text)
    if cfg.remove_hashtags:
        text = _remove_hashtags(text)
    if cfg.remove_emojis:
        text = _remove_emojis(text)

    # ── Step 5: Boilerplate removal (prose/web) ──
    if cfg.mode in ("prose", "social"):
        text = _remove_boilerplate(text)

    # ── Step 6: Whitespace normalisation ──────
    if cfg.normalize_spaces:
        text = _normalize_whitespace(text)

    # ── Step 7: Sentence splitting + filtering ──
    sentences = split_sentences(text, min_len=cfg.min_sentence_len)

    # ── Step 8: Deduplication ─────────────────
    duplicates_removed = 0
    if cfg.dedupe_lines and sentences:
        sentences, duplicates_removed = deduplicate(
            sentences, threshold=cfg.dedupe_threshold
        )

    # ── Step 9: Sentence cap ──────────────────
    sentences = sentences[:cfg.max_sentences]
    text = " ".join(sentences)

    # ── Step 10: Token budget ─────────────────
    truncated = False
    if cfg.max_tokens:
        text, truncated = trim_to_token_budget(text, cfg.max_tokens)

    return CleanResult(
        text=text,
        mode=cfg.mode,
        original_chars=original_chars,
        cleaned_chars=len(text),
        sentences=len(sentences),
        duplicates_removed=duplicates_removed,
        truncated=truncated,
    )


# ─────────────────────────────────────────────
# CONVENIENCE WRAPPER
# ─────────────────────────────────────────────

def clean(
    text: str,
    mode: str = "prose",
    max_tokens: Optional[int] = None,
    **overrides,
) -> CleanResult:
    """
    Simple interface. Pass mode="transcript" | "code" | "ocr" | "social" | "prose".

    Example:
        result = clean(raw_text, mode="transcript", max_tokens=3000)
        print(result.text)
    """
    cfg = CleanConfig(**{
        **CONFIGS.get(mode, CONFIGS["prose"]).__dict__,
        **({"max_tokens": max_tokens} if max_tokens else {}),
        **overrides,
    })
    return clean_text(text, config=cfg)


# ─────────────────────────────────────────────
# COMMENTS CLEANER
# ─────────────────────────────────────────────

def clean_comments(
    comments: list[str],
    config: CleanConfig | None = None,
) -> list[str]:
    """
    Clean and deduplicate a list of social comments.
    Returns filtered list of clean comment strings.
    """
    cfg = config or CONFIGS["social"]
    cleaned = []

    for c in comments:
        if not c or not isinstance(c, str):
            continue
        result = clean_text(c.strip(), config=cfg)
        if result.cleaned_chars >= cfg.min_comment_len:
            cleaned.append(result.text)

    deduped, _ = deduplicate(cleaned, threshold=cfg.dedupe_threshold)
    return deduped[:cfg.max_comments]


# ─────────────────────────────────────────────
# PROCESSOR OUTPUT CLEANER
# ─────────────────────────────────────────────

# Maps processor output field names → clean mode
_FIELD_MODES: dict[str, str] = {
    "content":          "prose",
    "transcript":       "transcript",
    "caption":          "social",
    "overview":         "prose",
    "description":      "prose",
    "readme":           "prose",
    "ocr_text":         "ocr",
    "code":             "code",
    "comments":         "social",
    "unique_comments":  "social",
    "summary":          "prose",
    "article":          "prose",
    "text":             "prose",
    "body":             "prose",
}

def clean_processor_output(
    data: dict,
    max_tokens: Optional[int] = None,
) -> dict:
    """
    Takes raw output dict from any processor and returns a cleaned,
    LLM-ready dict. Preserves all keys; cleans values based on field type.

    Args:
        data:       Raw processor output dict
        max_tokens: Optional per-field token budget

    Returns:
        Dict with same keys, cleaned string values
    """
    result = {}

    for key, value in data.items():
        if not value:
            continue

        # List fields (comments)
        if isinstance(value, list):
            if key in ("comments", "unique_comments"):
                result[key] = clean_comments(value)
            else:
                # List of strings — join and clean as prose
                joined = "\n".join(str(v) for v in value if v)
                r = clean(joined, mode="prose", max_tokens=max_tokens)
                result[key] = r.text

        elif isinstance(value, str):
            mode = _FIELD_MODES.get(key, "prose")
            r = clean(value, mode=mode, max_tokens=max_tokens)
            result[key] = r.text

        elif isinstance(value, dict):
            # Nested dict — recurse
            result[key] = clean_processor_output(value, max_tokens=max_tokens)

        else:
            # Numbers, booleans, etc. — pass through unchanged
            result[key] = value

    return result


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Clean text for LLM input")
    parser.add_argument("file",        help="Text file to clean")
    parser.add_argument("--mode",      default="prose",
                        choices=list(CONFIGS.keys()),
                        help="Cleaning mode (default: prose)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Token budget limit")
    parser.add_argument("--stats",     action="store_true",
                        help="Show cleaning statistics")
    args = parser.parse_args()

    raw = open(args.file, encoding="utf-8").read()
    result = clean(raw, mode=args.mode, max_tokens=args.max_tokens)

    print(result.text)

    if args.stats:
        print(f"\n── Stats ──────────────────────────")
        print(f"Mode            : {result.mode}")
        print(f"Original chars  : {result.original_chars:,}")
        print(f"Cleaned chars   : {result.cleaned_chars:,}")
        print(f"Compression     : {result.compression_ratio:.1%}")
        print(f"Sentences kept  : {result.sentences}")
        print(f"Duplicates rm'd : {result.duplicates_removed}")
        print(f"Truncated       : {result.truncated}")
