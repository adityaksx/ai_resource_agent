"""
utils/cleaner.py
----------------
Cleans and normalises raw text extracted from any source before
it is passed to the LLM.

Responsibilities (this file only):
  - Clean raw text strings (unicode, HTML, boilerplate, whitespace)
  - Deduplicate near-identical sentences
  - Split text into sentences
  - Trim to token budget
  - Clean processor output dicts field-by-field

Does NOT:
  - Build LLM prompts         → llm/prompt_builder.py
  - Call the LLM              → llm/summarizer.py
  - Detect source type        → utils/source_detector.py
  - Run multi-stage pipeline  → llm/pipeline.py
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CleanConfig:
    remove_urls:      bool          = True
    remove_hashtags:  bool          = True
    remove_emojis:    bool          = False   # keep by default — useful context
    remove_html_tags: bool          = True
    fix_unicode:      bool          = True
    normalize_spaces: bool          = True
    normalize_quotes: bool          = True
    dedupe_lines:     bool          = True
    dedupe_threshold: float         = 0.85    # Jaccard similarity threshold
    min_sentence_len: int           = 15      # chars — shorter sentences dropped
    max_sentences:    int           = 100
    max_comments:     int           = 30
    min_comment_len:  int           = 10
    max_tokens:       Optional[int] = None    # rough: 1 token ≈ 4 chars
    mode:             str           = "prose" # prose | transcript | code | ocr | social


# Preset configs — used by clean(mode=...) and clean_processor_output()
CONFIGS: dict[str, CleanConfig] = {
    "prose":      CleanConfig(mode="prose"),
    "transcript": CleanConfig(mode="transcript", remove_urls=True,
                              remove_hashtags=False, max_sentences=200),
    "code":       CleanConfig(mode="code", remove_urls=False,
                              remove_hashtags=False, remove_emojis=False,
                              dedupe_lines=False),
    "ocr":        CleanConfig(mode="ocr", remove_urls=True,
                              min_sentence_len=5, dedupe_threshold=0.9),
    "social":     CleanConfig(mode="social", remove_urls=True,
                              remove_hashtags=True, remove_emojis=True,
                              max_sentences=50),
}


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CleanResult:
    text:               str
    mode:               str
    original_chars:     int
    cleaned_chars:      int
    sentences:          int
    duplicates_removed: int  = 0
    truncated:          bool = False

    @property
    def compression_ratio(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return round(1 - self.cleaned_chars / self.original_chars, 3)

    def __str__(self) -> str:
        return self.text


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL CLEANERS  (private — use clean() or clean_text() externally)
# ─────────────────────────────────────────────────────────────────────────────

def _fix_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def _normalize_quotes(text: str) -> str:
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--",
        "\u2026": "...",
    }
    for smart, straight in replacements.items():
        text = text.replace(smart, straight)
    return text


def _remove_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    entities = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&apos;": "'", "&nbsp;": " ", "&#39;": "'",
    }
    for ent, char in entities.items():
        text = text.replace(ent, char)
    return text


def _remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def _remove_hashtags(text: str) -> str:
    return re.sub(r"#\w+", "", text)


def _remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U00002600-\U000026FF"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _remove_vtt_artifacts(text: str) -> str:
    text = re.sub(
        r"\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}", "", text
    )
    text = re.sub(r"^\d+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^WEBVTT.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", "", text)
    return text


def _remove_boilerplate(text: str) -> str:
    patterns = [
        r"cookie[s]? policy", r"accept all cookies", r"privacy policy",
        r"terms of (service|use)", r"all rights reserved",
        r"subscribe (to our newsletter|now)",
        r"click here to (read|view|see)", r"advertisement",
        r"skip to (main )?content", r"share this (article|post|story)",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
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
    Public — can be imported by other modules if needed.
    """
    seen:   list[str] = []
    result: list[str] = []
    removed = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append(line)
            continue
        if any(_jaccard(stripped, s) >= threshold for s in seen):
            removed += 1
        else:
            seen.append(stripped)
            result.append(line)

    return result, removed


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE SPLITTING
# ─────────────────────────────────────────────────────────────────────────────

_ABBREVS = [
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr",
    "vs", "etc", "approx", "est", "vol", "fig",
    "no", "p", "pp", "e.g", "i.e", "Jan", "Feb",
    "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct",
    "Nov", "Dec", "U.S", "U.K",
]
_DOT_PLACEHOLDER = "<DOT>"   # must be non-empty — empty string caused restore bug


def split_sentences(text: str, min_len: int = 15) -> list[str]:
    """
    Split text into sentences, protecting common abbreviations.
    Public — used by clean_text() and can be imported externally.
    """
    if not text:
        return []

    protected = text
    for abbr in _ABBREVS:
        protected = protected.replace(f"{abbr}.", f"{abbr}{_DOT_PLACEHOLDER}")

    parts     = re.split(r"(?<=[.!?])\s+", protected)
    sentences = []
    for part in parts:
        restored = part.replace(_DOT_PLACEHOLDER, ".").strip()
        if len(restored) >= min_len:
            sentences.append(restored)

    return sentences


# ─────────────────────────────────────────────────────────────────────────────
# TOKEN BUDGET
# ─────────────────────────────────────────────────────────────────────────────

def trim_to_token_budget(text: str, max_tokens: int) -> tuple[str, bool]:
    """
    Trim text to approximately max_tokens (1 token ≈ 4 chars).
    Trims at sentence boundary where possible.
    Returns (trimmed_text, was_truncated).
    Public — summarizer.py can import this if needed.
    """
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text, False

    chunk      = text[:max_chars]
    last_break = max(chunk.rfind(". "), chunk.rfind(".\n"))
    if last_break > max_chars * 0.7:
        chunk = chunk[:last_break + 1]

    return chunk.strip(), True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLEAN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str, config: CleanConfig | None = None) -> CleanResult:
    """
    Full cleaning pipeline for a single text string.
    Returns a CleanResult dataclass (str-compatible via __str__).
    """
    if not text or not isinstance(text, str):
        return CleanResult(text="", mode="prose",
                           original_chars=0, cleaned_chars=0, sentences=0)

    cfg            = config or CleanConfig()
    original_chars = len(text)

    # Step 1: Unicode + encoding
    if cfg.fix_unicode:
        text = _fix_unicode(text)
    if cfg.normalize_quotes:
        text = _normalize_quotes(text)

    # Step 2: Format-specific pre-cleaning
    if cfg.mode == "transcript":
        text = _remove_vtt_artifacts(text)
    if cfg.remove_html_tags:
        text = _remove_html(text)

    # Step 3: Code mode — skip destructive cleaning, return early
    if cfg.mode == "code":
        if cfg.normalize_spaces:
            text = _normalize_whitespace(text)
        return CleanResult(
            text=text, mode=cfg.mode,
            original_chars=original_chars, cleaned_chars=len(text),
            sentences=len(text.splitlines()),
        )

    # Step 4: Content removal
    if cfg.remove_urls:
        text = _remove_urls(text)
    if cfg.remove_hashtags:
        text = _remove_hashtags(text)
    if cfg.remove_emojis:
        text = _remove_emojis(text)

    # Step 5: Boilerplate (prose + social only)
    if cfg.mode in ("prose", "social"):
        text = _remove_boilerplate(text)

    # Step 6: Whitespace
    if cfg.normalize_spaces:
        text = _normalize_whitespace(text)

    # Step 7: Sentence split + length filter
    sentences = split_sentences(text, min_len=cfg.min_sentence_len)

    # Step 8: Deduplication
    duplicates_removed = 0
    if cfg.dedupe_lines and sentences:
        sentences, duplicates_removed = deduplicate(
            sentences, threshold=cfg.dedupe_threshold
        )

    # Step 9: Sentence cap
    sentences = sentences[:cfg.max_sentences]
    text      = " ".join(sentences)

    # Step 10: Token budget
    truncated = False
    if cfg.max_tokens:
        text, truncated = trim_to_token_budget(text, cfg.max_tokens)

    return CleanResult(
        text=text, mode=cfg.mode,
        original_chars=original_chars, cleaned_chars=len(text),
        sentences=len(sentences),
        duplicates_removed=duplicates_removed,
        truncated=truncated,
    )


def clean(
    text:       str,
    mode:       str          = "prose",
    max_tokens: Optional[int] = None,
    **overrides,
) -> CleanResult:
    """
    Convenience wrapper. Use mode= to select preset config.
    Extra kwargs override individual CleanConfig fields.

    Usage:
        result = clean(raw_text, mode="transcript", max_tokens=3000)
        print(result.text)
    """
    cfg = CleanConfig(**{
        **CONFIGS.get(mode, CONFIGS["prose"]).__dict__,
        **({"max_tokens": max_tokens} if max_tokens else {}),
        **overrides,
    })
    return clean_text(text, config=cfg)


# ─────────────────────────────────────────────────────────────────────────────
# COMMENTS CLEANER
# ─────────────────────────────────────────────────────────────────────────────

def clean_comments(
    comments: list[str],
    config:   CleanConfig | None = None,
) -> list[str]:
    """
    Clean and deduplicate a list of social media comments.
    Returns filtered list of clean comment strings.
    Called by clean_processor_output() for 'comments' / 'unique_comments' fields.
    """
    cfg     = config or CONFIGS["social"]
    cleaned = []

    for c in comments:
        if not c or not isinstance(c, str):
            continue
        result = clean_text(c.strip(), config=cfg)
        if result.cleaned_chars >= cfg.min_comment_len:
            cleaned.append(result.text)

    deduped, _ = deduplicate(cleaned, threshold=cfg.dedupe_threshold)
    return deduped[:cfg.max_comments]


# ─────────────────────────────────────────────────────────────────────────────
# PROCESSOR OUTPUT CLEANER  ← main entry point called by main.py
# ─────────────────────────────────────────────────────────────────────────────

# Metadata fields — pass through raw, NEVER clean these.
# Bug fixed: source_type="github_repo" (10 chars) was being dropped by
# min_sentence_len=15, breaking model routing in summarizer.py.
_SKIP_CLEAN_FIELDS: set[str] = {
    "source_type", "url", "video_id", "channel",
    "author", "date", "source", "title",
}

# Content fields → which clean mode to use
_FIELD_MODES: dict[str, str] = {
    "content":         "prose",
    "transcript":      "transcript",
    "caption":         "social",
    "overview":        "prose",
    "description":     "prose",
    "readme":          "prose",
    "ocr_text":        "ocr",
    "code":            "code",
    "comments":        "social",
    "unique_comments": "social",
    "summary":         "prose",
    "article":         "prose",
    "text":            "prose",
    "body":            "prose",
}


def clean_processor_output(
    data:       dict,
    max_tokens: Optional[int] = None,
) -> dict:
    """
    Takes raw output dict from any processor and returns a cleaned,
    LLM-ready dict.

    Rules:
    - Metadata fields (_SKIP_CLEAN_FIELDS) → pass through unchanged
    - String fields  → cleaned with mode from _FIELD_MODES
    - List[str]      → comments cleaned with clean_comments(), others joined+prose
    - Dict fields    → recursively cleaned
    - Numbers/bools  → passed through unchanged

    Called by: main.py (process_link, process_text_input, process_image_input)
    NOT called by: prompt_builder, summarizer, pipeline — they receive already-cleaned dicts
    """
    result: dict = {}

    for key, value in data.items():
        if not value and value != 0:   # keep 0 but skip None/""/[]
            continue

        # ── Metadata: never clean ─────────────────────────────────────────
        if key in _SKIP_CLEAN_FIELDS:
            result[key] = value
            continue

        # ── Lists ─────────────────────────────────────────────────────────
        if isinstance(value, list):
            if key in ("comments", "unique_comments"):
                result[key] = clean_comments(value)
            else:
                joined      = "\n".join(str(v) for v in value if v)
                r           = clean(joined, mode="prose", max_tokens=max_tokens)
                result[key] = r.text

        # ── Strings ───────────────────────────────────────────────────────
        elif isinstance(value, str):
            mode        = _FIELD_MODES.get(key, "prose")
            r           = clean(value, mode=mode, max_tokens=max_tokens)
            result[key] = r.text

        # ── Nested dicts ──────────────────────────────────────────────────
        elif isinstance(value, dict):
            result[key] = clean_processor_output(value, max_tokens=max_tokens)

        # ── Numbers, booleans ─────────────────────────────────────────────
        else:
            result[key] = value

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI  (python -m utils.cleaner myfile.txt --mode transcript --stats)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Clean text for LLM input")
    parser.add_argument("file", help="Text file to clean")
    parser.add_argument("--mode", default="prose",
                        choices=list(CONFIGS.keys()),
                        help="Cleaning mode (default: prose)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Token budget limit")
    parser.add_argument("--stats", action="store_true",
                        help="Show cleaning statistics")
    args = parser.parse_args()

    raw    = open(args.file, encoding="utf-8").read()
    result = clean(raw, mode=args.mode, max_tokens=args.max_tokens)

    print(result.text)

    if args.stats:
        print(f"\n── Stats ──────────────────────────")
        print(f"Mode             : {result.mode}")
        print(f"Original chars   : {result.original_chars:,}")
        print(f"Cleaned chars    : {result.cleaned_chars:,}")
        print(f"Compression      : {result.compression_ratio:.1%}")
        print(f"Sentences kept   : {result.sentences}")
        print(f"Duplicates rm'd  : {result.duplicates_removed}")
        print(f"Truncated        : {result.truncated}")
