"""
utils/source_detector.py
------------------------
Detects the source type from:
  - URLs (YouTube, Instagram, GitHub, Twitter/X, LinkedIn, Reddit,
    Notion, Medium, Substack, HuggingFace, ArXiv, Pastebin, Loom, Vimeo, generic web)
  - Local file paths (image, video, audio, document, code, data, archive)
  - Raw text / plain notes (code snippet, JSON, markdown, plain text)

Returns a fine-grained string tag consumed by:
  - main.py          → routing + _UNSUPPORTED_SOURCES lookup
  - llm/pipeline.py  → extract_guidance(), model selection
  - llm/summarizer.py → get_model()

Does NOT:
  - Call the LLM         → llm/llm_classifier.py (Stage 1 fallback)
  - Clean text           → utils/cleaner.py
  - Build prompts        → llm/prompt_builder.py

Return value contract (all possible values):
  YouTube   : youtube_video, youtube_shorts, youtube_playlist,
              youtube_channel, youtube_live
  Instagram : instagram_reel, instagram_post, instagram_story,
              instagram_tv, instagram_profile
  GitHub    : github_repo, github_user, github_gist, github_file
  Twitter   : twitter_tweet, twitter_profile
  LinkedIn  : linkedin_post, linkedin_article, linkedin_company,
              linkedin_profile
  Reddit    : reddit_post, reddit_subreddit
  HuggingFace: huggingface_model, huggingface_dataset, huggingface_space
  Research  : arxiv_paper
  Publishing: medium_article, substack_article, notion_page, pastebin
  Video     : loom_video, vimeo_video, video_url
  URL files : image_url, pdf_url, audio_url, code_url, data_url
  Generic   : web
  Local     : local_image, local_video, local_audio,
              pdf_document, word_document, spreadsheet,
              presentation, ebook, code_file, notebook,
              data_file, archive, plain_text_file
  Raw text  : code_snippet, json_data, markdown, plain_text
  Fallback  : unknown
"""

from __future__ import annotations

import os
import re
import json
from urllib.parse import urlparse, parse_qs


# ─────────────────────────────────────────────────────────────────────────────
# FILE EXTENSION MAPS
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp",
                 ".tiff", ".tif", ".svg", ".ico", ".heic", ".avif"}
VIDEO_EXTS    = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv",
                 ".wmv", ".m4v", ".3gp", ".ogv"}
AUDIO_EXTS    = {".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a",
                 ".wma", ".opus", ".aiff"}
DOCUMENT_EXTS = {".pdf", ".docx", ".doc", ".odt", ".rtf",
                 ".pptx", ".ppt", ".odp",
                 ".xlsx", ".xls", ".ods", ".csv",
                 ".epub", ".mobi"}
CODE_EXTS     = {".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".htm",
                 ".css", ".scss", ".sass", ".java", ".kt", ".c", ".cpp",
                 ".h", ".cs", ".go", ".rs", ".rb", ".php", ".swift",
                 ".sh", ".bash", ".zsh", ".ps1", ".r", ".m", ".sql",
                 ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
                 ".dockerfile", ".makefile"}
DATA_EXTS     = {".json", ".xml", ".ndjson", ".jsonl", ".parquet",
                 ".arrow", ".hdf5", ".h5", ".feather", ".db", ".sqlite"}
ARCHIVE_EXTS  = {".zip", ".tar", ".gz", ".bz2", ".xz", ".rar",
                 ".7z", ".tgz"}
NOTEBOOK_EXTS = {".ipynb"}
TEXT_EXTS     = {".txt", ".md", ".rst", ".log"}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def detect_source(input_str: str) -> str:
    """
    Detect source type of any input string.

    Priority:
      1. Local file path  (exists on disk OR has path separators)
      2. URL              (starts with http:// or https://)
      3. Raw text         (code snippet, JSON, markdown, plain text)
    """
    if not input_str or not isinstance(input_str, str):
        return "unknown"

    s = input_str.strip()

    # 1. Local file
    local_type = _detect_local_file(s)
    if local_type:
        return local_type

    # 2. URL
    if re.match(r"^https?://", s, re.IGNORECASE):
        return _detect_url(s)

    # 3. Raw text
    return _detect_raw_text(s)


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL FILE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _looks_like_path(s: str) -> bool:
    """
    True only when the string is structurally a file path.
    Prevents bare filenames like "photo.jpg" in pasted text from
    being misrouted to the image processor.
    """
    if os.path.exists(s):
        return True
    if os.sep in s or "/" in s:
        return True
    if re.match(r"^[A-Za-z]:[/\\]", s):     # Windows: C:\ or D:/
        return True
    if s.startswith("~") or s.startswith("/"):
        return True
    return False


def _detect_local_file(path: str) -> str | None:
    """
    Returns a type tag if the input is a local file path, else None.
    Never classifies HTTP URLs as local files.
    """
    if re.match(r"^https?://", path, re.IGNORECASE):
        return None
    if not _looks_like_path(path):
        return None

    # Multi-part extension must be checked before splitext
    if path.lower().endswith(".tar.gz"):
        return "archive"

    _, ext = os.path.splitext(path.lower())

    if not ext:
        return "plain_text_file" if os.path.isfile(path) else None

    if ext in IMAGE_EXTS:    return "local_image"
    if ext in VIDEO_EXTS:    return "local_video"
    if ext in AUDIO_EXTS:    return "local_audio"
    if ext in NOTEBOOK_EXTS: return "notebook"
    if ext in CODE_EXTS:     return "code_file"
    if ext in DATA_EXTS:     return "data_file"
    if ext in ARCHIVE_EXTS:  return "archive"
    if ext in TEXT_EXTS:     return "plain_text_file"

    if ext == ".pdf":                             return "pdf_document"
    if ext in {".docx", ".doc", ".odt", ".rtf"}: return "word_document"
    if ext in {".xlsx", ".xls", ".ods", ".csv"}: return "spreadsheet"
    if ext in {".pptx", ".ppt", ".odp"}:         return "presentation"
    if ext in {".epub", ".mobi"}:                 return "ebook"

    return "plain_text_file" if os.path.isfile(path) else None


# ─────────────────────────────────────────────────────────────────────────────
# URL DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_url(url: str) -> str:
    try:
        parsed = urlparse(url.strip())
        domain = parsed.netloc.lower().lstrip("www.")
        path   = parsed.path.lower()
        query  = parse_qs(parsed.query)

        # ── YouTube ──────────────────────────────────────────────────────
        if "youtube.com" in domain or "youtu.be" in domain:
            if "playlist" in query or "playlist" in path:
                return "youtube_playlist"
            if "/shorts/" in path:
                return "youtube_shorts"
            if ("/channel/" in path or "/c/" in path
                    or "/user/" in path or "/@" in path):
                return "youtube_channel"
            if "/live/" in path or "live" in query:
                return "youtube_live"
            return "youtube_video"

        # ── Instagram ────────────────────────────────────────────────────
        if "instagram.com" in domain:
            if "/reel/"    in path: return "instagram_reel"
            if "/p/"       in path: return "instagram_post"
            if "/stories/" in path: return "instagram_story"
            if "/tv/"      in path: return "instagram_tv"
            return "instagram_profile"          # profile — blocked in main.py

        # ── GitHub ───────────────────────────────────────────────────────
        if "github.com" in domain:
            # gist.github.com must be checked BEFORE path-part logic
            if "gist.github.com" in domain:
                return "github_gist"
            parts = [p for p in path.split("/") if p]
            if len(parts) >= 3 and parts[2] in ("blob", "tree", "raw", "edit"):
                return "github_file"
            if len(parts) >= 2:
                return "github_repo"
            # FIX: was "github_profile" — changed to "github_user" to match
            # main.py _UNSUPPORTED_SOURCES key so friendly message is shown
            return "github_user"

        # ── Twitter / X ──────────────────────────────────────────────────
        if "twitter.com" in domain or "x.com" in domain:
            if "/status/" in path: return "twitter_tweet"
            return "twitter_profile"

        # ── LinkedIn ─────────────────────────────────────────────────────
        if "linkedin.com" in domain:
            if "/posts/"   in path: return "linkedin_post"
            if "/pulse/"   in path: return "linkedin_article"
            if "/company/" in path: return "linkedin_company"
            if "/in/"      in path: return "linkedin_profile"   # blocked in main.py
            return "web"

        # ── Reddit ───────────────────────────────────────────────────────
        if "reddit.com" in domain:
            if "/comments/" in path: return "reddit_post"
            if "/r/"        in path: return "reddit_subreddit"
            return "web"

        # ── HuggingFace ──────────────────────────────────────────────────
        if "huggingface.co" in domain:
            if "/datasets/" in path: return "huggingface_dataset"
            if "/spaces/"   in path: return "huggingface_space"
            return "huggingface_model"

        # ── ArXiv ────────────────────────────────────────────────────────
        if "arxiv.org" in domain:
            return "arxiv_paper"

        # ── Medium ───────────────────────────────────────────────────────
        if "medium.com" in domain:
            return "medium_article"

        # ── Substack ─────────────────────────────────────────────────────
        if "substack.com" in domain:
            return "substack_article"

        # ── Notion ───────────────────────────────────────────────────────
        if "notion.so" in domain or "notion.site" in domain:
            return "notion_page"

        # ── Pastebin ─────────────────────────────────────────────────────
        # NOTE: gist.github.com is handled above in the GitHub block.
        # Only pastebin.com lands here.
        if "pastebin.com" in domain:
            return "pastebin"

        # ── Video platforms ──────────────────────────────────────────────
        if "loom.com"        in domain: return "loom_video"
        if "vimeo.com"       in domain: return "vimeo_video"
        if "dailymotion.com" in domain: return "video_url"
        if "tiktok.com"      in domain: return "video_url"

        # ── URL pointing to a known file type ────────────────────────────
        path_ext = os.path.splitext(path)[-1].lower()
        if path_ext in IMAGE_EXTS: return "image_url"
        if path_ext in VIDEO_EXTS: return "video_url"
        if path_ext in AUDIO_EXTS: return "audio_url"
        if path_ext == ".pdf":     return "pdf_url"
        if path_ext in CODE_EXTS:  return "code_url"
        if path_ext in DATA_EXTS:  return "data_url"

        return "web"

    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# RAW TEXT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_raw_text(text: str) -> str:
    """
    Classify raw text that is not a URL or local file path.
    Used when user pastes a code snippet, JSON blob, markdown note, etc.
    The LLM classifier in llm/llm_classifier.py will refine this further.
    """
    if not text:
        return "unknown"

    stripped = text.strip()

    # ── JSON ─────────────────────────────────────────────────────────────
    if ((stripped.startswith("{") and stripped.endswith("}")) or
            (stripped.startswith("[") and stripped.endswith("]"))):
        try:
            json.loads(stripped)
            return "json_data"
        except Exception:
            pass    # malformed JSON — fall through to other checks

    # ── Code patterns ────────────────────────────────────────────────────
    code_patterns = [
        # Python
        r"^\s*(def |class |import |from |async def |@)",
        r"^\s*if __name__\s*==",
        # JavaScript / TypeScript
        r"^\s*(function |const |let |var |=>|export |import )",
        r"^\s*(async function|await )",
        # Shell
        r"^#!/",
        r"^\s*(echo |cd |ls |mkdir |chmod |sudo )",
        # C / C++ / Java / Go / Rust
        r"^\s*(int |void |public |private |protected |class |struct |fn |func )",
        r"^\s*#include\s*<",
        r"^\s*package\s+\w+",
        # SQL
        r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\s",
        # Generic: indented block OR multiple lines with = assignments
        r"^\s{4,}\w",
    ]
    for pat in code_patterns:
        if re.search(pat, stripped, re.IGNORECASE | re.MULTILINE):
            return "code_snippet"

    # ── Markdown ─────────────────────────────────────────────────────────
    markdown_patterns = [
        r"^#{1,6}\s+\w",            # headings
        r"^\s*[-*]\s+\w",           # unordered list
        r"^\s*\d+\.\s+\w",          # ordered list
        r"\*\*.+?\*\*",             # bold
        r"`.+?`",                   # inline code
        r"^\s*```",                 # fenced code block
        r"\[.+?\]\(.+?\)",          # links
        r"^\s*>\s+\w",              # blockquote
        r"^\s*---\s*$",             # horizontal rule
    ]
    markdown_hits = sum(
        1 for pat in markdown_patterns
        if re.search(pat, stripped, re.MULTILINE)
    )
    if markdown_hits >= 2:
        return "markdown"

    return "plain_text"


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD FILE DETECTOR  — for FastAPI UploadFile objects
# ─────────────────────────────────────────────────────────────────────────────

def detect_upload(filename: str, mime_type: str | None = None) -> str:
    """
    Classify an uploaded file by filename and optional MIME type.

    Usage in web/app.py:
        source = detect_upload(upload_file.filename, upload_file.content_type)
    """
    if not filename:
        return "unknown"

    # MIME type takes priority over extension
    if mime_type:
        m = mime_type.lower()
        if m.startswith("image/"):              return "local_image"
        if m.startswith("video/"):              return "local_video"
        if m.startswith("audio/"):              return "local_audio"
        if m == "application/pdf":             return "pdf_document"
        if "spreadsheet" in m:                 return "spreadsheet"
        if "presentation" in m:               return "presentation"
        if "wordprocessing" in m:             return "word_document"
        if m in ("application/json",
                 "application/xml",
                 "text/csv"):                  return "data_file"
        if m == "text/plain":                  return "plain_text_file"
        if m in ("application/zip",
                 "application/x-tar"):         return "archive"

    # Fall back to file extension
    # Note: _looks_like_path() is NOT used here because uploaded filenames
    # don't have directory separators but are still real files.
    _, ext = os.path.splitext(filename.lower())

    if ext in IMAGE_EXTS:    return "local_image"
    if ext in VIDEO_EXTS:    return "local_video"
    if ext in AUDIO_EXTS:    return "local_audio"
    if ext in NOTEBOOK_EXTS: return "notebook"
    if ext in CODE_EXTS:     return "code_file"
    if ext in DATA_EXTS:     return "data_file"
    if ext in ARCHIVE_EXTS:  return "archive"
    if ext in TEXT_EXTS:     return "plain_text_file"

    if ext == ".pdf":                             return "pdf_document"
    if ext in {".docx", ".doc", ".odt", ".rtf"}: return "word_document"
    if ext in {".xlsx", ".xls", ".ods", ".csv"}: return "spreadsheet"
    if ext in {".pptx", ".ppt", ".odp"}:         return "presentation"
    if ext in {".epub", ".mobi"}:                 return "ebook"

    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# CLI TEST  (python -m utils.source_detector)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # YouTube
        ("https://youtube.com/watch?v=abc",           "youtube_video"),
        ("https://youtube.com/playlist?list=123",     "youtube_playlist"),
        ("https://youtube.com/shorts/abc123",         "youtube_shorts"),
        ("https://youtube.com/@fireship",             "youtube_channel"),
        ("https://youtube.com/live/abc",              "youtube_live"),
        # Instagram
        ("https://instagram.com/p/abc",               "instagram_post"),
        ("https://instagram.com/reel/xyz",            "instagram_reel"),
        ("https://www.instagram.com/aditya.ksx/",    "instagram_profile"),
        # GitHub
        ("https://github.com/openai/gpt",             "github_repo"),
        ("https://github.com/torvalds/linux/blob/master/README", "github_file"),
        ("https://gist.github.com/user/abc123",       "github_gist"),
        ("https://github.com/palinkiewicz",           "github_user"),   # FIX: was github_profile
        # Twitter
        ("https://twitter.com/elonmusk/status/123",   "twitter_tweet"),
        # LinkedIn
        ("https://linkedin.com/in/johndoe",           "linkedin_profile"),
        ("https://linkedin.com/company/openai",       "linkedin_company"),
        # Reddit
        ("https://reddit.com/r/Python/comments/abc",  "reddit_post"),
        ("https://reddit.com/r/Python",               "reddit_subreddit"),
        # AI / Research
        ("https://huggingface.co/datasets/squad",     "huggingface_dataset"),
        ("https://arxiv.org/abs/2303.08774",          "arxiv_paper"),
        # Publishing
        ("https://medium.com/@user/article",          "medium_article"),
        ("https://mysite.substack.com/p/issue-1",     "substack_article"),
        # Video
        ("https://loom.com/share/abc123",             "loom_video"),
        ("https://vimeo.com/123456",                  "vimeo_video"),
        # Local paths
        ("/home/user/photo.jpg",                      "local_image"),
        ("D:\\Downloads\\report.pdf",                 "pdf_document"),
        ("C:/projects/script.py",                     "code_file"),
        # Bare filenames — must NOT be classified as local files
        ("photo.jpg",                                 "plain_text"),
        ("script.py",                                 "plain_text"),
        # Raw text
        ("def hello(): print('hi')",                  "code_snippet"),
        ('{"key": "value"}',                          "json_data"),
        ("# My Notes\nThis is **markdown**\n- item",  "markdown"),
        ("Just some plain text note here.",           "plain_text"),
        # Generic web
        ("https://example.com/article",              "web"),
        ("https://adityaksx.vercel.app/",            "web"),
    ]

    passed = 0
    failed = 0
    print(f"{'Input':<52} {'Expected':<22} {'Got':<22} {'✓/?'}")
    print("─" * 100)
    for inp, expected in tests:
        got  = detect_source(inp)
        ok   = "✅" if got == expected else "❌"
        if got == expected: passed += 1
        else:               failed += 1
        print(f"{inp[:51]:<52} {expected:<22} {got:<22} {ok}")

    print(f"\n{passed}/{passed+failed} passed")
