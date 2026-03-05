"""
main.py
-------
Central router for the AI Resource Agent.

Handles four input types:
  1. URL / link        → detect source → processor → clean → LLM
  2. Plain text / note → text_processor → clean → LLM
  3. Local image       → image_processor (OCR) → clean → LLM
  4. Local file        → appropriate processor → clean → LLM
  5. Mixed input       → all of the above combined in one call
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path

from utils.source_detector  import detect_source
from utils.cleaner          import clean_processor_output
from llm.prompt_builder     import build_summary_prompt
from llm.summarizer         import summarize
from database.db            import save_resource, init_db

# ── Processors ──────────────────────────────
from processors.youtube_processor   import process_youtube
from processors.github_processor    import process_github
from processors.web_processor       import process_web
from processors.instagram_processor import process_instagram
from processors.text_processor      import process_text
from processors.image_processor     import process_image

logger = logging.getLogger(__name__)

# Source types that should NEVER reach process_link
# (they are local files, handled by process_local_file)
_LOCAL_SOURCE_TYPES = {
    "local_image", "local_video", "local_audio",
    "pdf_document", "word_document", "spreadsheet",
    "presentation", "ebook", "code_file", "notebook",
    "data_file", "archive", "plain_text_file",
}

# Source types routed to web processor
_WEB_SOURCE_TYPES = {
    "web", "medium_article", "substack_article",
    "notion_page", "arxiv_paper", "reddit_post",
    "reddit_subreddit", "linkedin_post", "linkedin_article",
    "linkedin_company", "linkedin_profile",
    "huggingface_model", "huggingface_dataset", "huggingface_space",
    "pastebin", "loom_video", "vimeo_video",
    "image_url", "pdf_url", "audio_url", "code_url", "data_url",
    "video_url",
}


def _safe_json(obj) -> str:
    """Safely serialize any object to JSON string for DB storage."""
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def _save(
    source:       str,
    url:          str | None,
    title:        str,
    raw_input:    dict,
    raw_data:     dict | None = None,
    cleaned_data: dict | None = None,
    llm_output:   str  | None = None,
    status:       str         = "success",
    error:        str  | None = None,
):
    """Wrapper around save_resource with safe JSON serialization."""
    try:
        save_resource(
            source       = source,
            url          = url,
            title        = title,
            raw_input    = _safe_json(raw_input),
            raw_data     = _safe_json(raw_data)     if raw_data     else None,
            cleaned_data = _safe_json(cleaned_data) if cleaned_data else None,
            llm_output   = llm_output,
            status       = status,
            error        = error,
        )
    except Exception as e:
        logger.error(f"DB save failed: {e}")


# ─────────────────────────────────────────────
# SINGLE URL
# ─────────────────────────────────────────────

def process_link(url: str) -> str:
    """
    Process a single URL and return an LLM summary string.
    Detects source type and routes to the correct processor.
    """
    if not url or not url.strip():
        return "No URL provided."

    url    = url.strip()
    source = detect_source(url)

    # Guard: never route local file paths through this function
    if source in _LOCAL_SOURCE_TYPES:
        return f"'{url}' looks like a local file. Use process_image_input() or process_local_file() instead."

    raw_data = None

    try:
        # ── Route to processor ────────────────────
        if source.startswith("youtube"):
            raw_data = process_youtube(url)

        elif source.startswith("instagram"):
            raw_data = process_instagram(url)

        elif source.startswith("github"):
            raw_data = process_github(url)

        elif source in _WEB_SOURCE_TYPES or source == "unknown":
            raw_data = process_web(url)

        else:
            # Catch-all fallback
            raw_data = process_web(url)

        # Guard: processor returned None or empty
        if not raw_data:
            raise ValueError(f"Processor returned empty data for source '{source}'")

        # ── Clean → Prompt → LLM ──────────────────
        cleaned    = clean_processor_output(raw_data)
        prompt     = build_summary_prompt(cleaned)
        llm_output = summarize(prompt)

        _save(
            source       = source,
            url          = url,
            title        = raw_data.get("title", ""),
            raw_input    = {"url": url},
            raw_data     = raw_data,
            cleaned_data = cleaned,
            llm_output   = llm_output,
        )
        return llm_output

    except Exception as e:
        logger.error(f"process_link failed for {url}: {e}")
        _save(
            source    = source,
            url       = url,
            title     = "",
            raw_input = {"url": url},
            status    = "error",
            error     = str(e),
        )
        return f"Error processing link: {e}"


# ─────────────────────────────────────────────
# PLAIN TEXT
# ─────────────────────────────────────────────

def process_text_input(text: str) -> str:
    """
    Process plain user-typed or pasted text and return an LLM summary.
    Handles notes, code snippets, JSON, markdown etc.
    """
    if not text or not text.strip():
        return "No text provided."

    text = text.strip()

    try:
        raw_data   = process_text(text)
        cleaned    = clean_processor_output(raw_data)
        prompt     = build_summary_prompt(cleaned)
        llm_output = summarize(prompt)

        _save(
            source       = raw_data.get("source_type", "plain_text"),
            url          = None,
            title        = raw_data.get("title", text[:80]),
            raw_input    = {"text": text[:300]},
            raw_data     = raw_data,
            cleaned_data = cleaned,
            llm_output   = llm_output,
        )
        return llm_output

    except Exception as e:
        logger.error(f"process_text_input failed: {e}")
        return f"Error processing text: {e}"


# ─────────────────────────────────────────────
# IMAGE
# ─────────────────────────────────────────────

def process_image_input(image_path: str) -> str:
    """
    Run OCR on a local image file and return an LLM summary.
    Accepts absolute or relative paths.
    """
    image_path = str(Path(image_path).resolve())

    if not os.path.exists(image_path):
        return f"Image not found: {image_path}"

    try:
        raw_data   = process_image(image_path)
        cleaned    = clean_processor_output(raw_data)
        prompt     = build_summary_prompt(cleaned)
        llm_output = summarize(prompt)

        _save(
            source       = "local_image",
            url          = None,
            title        = raw_data.get("title", Path(image_path).name),
            raw_input    = {"image_path": image_path},
            raw_data     = raw_data,
            cleaned_data = cleaned,
            llm_output   = llm_output,
        )
        return llm_output

    except Exception as e:
        logger.error(f"process_image_input failed for {image_path}: {e}")
        return f"Error processing image '{Path(image_path).name}': {e}"


# ─────────────────────────────────────────────
# LOCAL FILE (PDF, code, audio, etc.)
# ─────────────────────────────────────────────

def process_local_file(file_path: str) -> str:
    """
    Process a local non-image file.
    Currently routes:
      - local_image        → process_image_input (OCR)
      - plain_text_file /
        code_file /
        notebook           → process_text_input (read + summarise)
      - others             → read as text best-effort
    """
    file_path = str(Path(file_path).resolve())

    if not os.path.exists(file_path):
        return f"File not found: {file_path}"

    source = detect_source(file_path)

    if source == "local_image":
        return process_image_input(file_path)

    if source in ("plain_text_file", "code_file", "notebook",
                  "data_file", "pdf_document"):
        try:
            text = Path(file_path).read_text(encoding="utf-8", errors="replace")
            return process_text_input(text)
        except Exception as e:
            return f"Error reading file '{Path(file_path).name}': {e}"

    return f"Unsupported local file type: {source} ({Path(file_path).name})"


# ─────────────────────────────────────────────
# MIXED INPUT  ← called by FastAPI /chat
# ─────────────────────────────────────────────

def process_input(
    text:        str       = "",
    image_paths: list[str] = None,
) -> str:
    """
    Main entry point for the web UI.

    Accepts any combination of:
      - Free text  (may include multiple URLs on separate lines)
      - Uploaded image file paths

    Returns a single combined LLM response string.
    """
    image_paths = [p for p in (image_paths or []) if p]
    parts       = []

    # ── 1. Split text into URLs vs plain text ──
    lines     = [l.strip() for l in (text or "").splitlines() if l.strip()]
    urls      = [
        l for l in lines
        if (l.startswith("http://") or l.startswith("https://"))
           and not os.path.exists(l)   # never treat a local path as URL
    ]
    plain     = [l for l in lines if l not in urls]
    plain_str = "\n".join(plain).strip()

    # ── 2. Process each URL ──────────────────────
    for url in urls:
        try:
            result = process_link(url)
            parts.append(f"[{url}]\n{result}")
        except Exception as e:
            parts.append(f"[{url}]\nError: {e}")

    # ── 3. Process plain text ────────────────────
    if plain_str:
        try:
            result = process_text_input(plain_str)
            parts.append(f"[Text Note]\n{result}")
        except Exception as e:
            parts.append(f"[Text Note]\nError: {e}")

    # ── 4. Process uploaded images ───────────────
    for img_path in image_paths:
        fname = Path(img_path).name
        try:
            if not os.path.exists(img_path):
                parts.append(f"[Image: {fname}]\nError: file not found.")
                continue
            result = process_image_input(img_path)
            parts.append(f"[Image: {fname}]\n{result}")
        except Exception as e:
            parts.append(f"[Image: {fname}]\nError: {e}")

    # ── 5. Nothing provided ──────────────────────
    if not parts:
        return "Nothing to process. Please provide a link, text, or image."

    # ── 6. Single input — return directly ────────
    if len(parts) == 1:
        return parts[0].split("\n", 1)[-1].strip()

    # ── 7. Multiple inputs — unify with LLM ──────
    combined_prompt = (
        "The user provided multiple resources. "
        "Summarise each one briefly, then give a combined insight:\n\n"
        + "\n\n---\n\n".join(parts)
    )
    try:
        return summarize(combined_prompt)
    except Exception:
        # Fallback: join raw parts if LLM fails
        return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────
# CLI ENTRY
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    init_db()

    print("AI Resource Agent")
    print("─" * 40)
    print("Enter a URL, paste text, or type a local file/image path.")
    print("Type 'exit' to quit.\n")

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}

    while True:
        try:
            user_input = input("Input: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            sys.exit(0)

        if not user_input:
            continue
        if user_input.lower() == "exit":
            break

        # Auto-detect input type
        p = Path(user_input)
        if p.exists() and p.suffix.lower() in IMAGE_EXTS:
            result = process_image_input(user_input)
        elif p.exists() and p.is_file():
            result = process_local_file(user_input)
        else:
            result = process_input(text=user_input)

        print(f"\n{result}\n")
