"""
main.py
-------
Central router for the AI Resource Agent.

Responsibilities (this file ONLY):
  - Accept input (URL / text / image / local file / mixed)
  - Detect source type (rule-based first, LLM classifier fallback)
  - Route to correct processor
  - Run clean → enrich → summarize pipeline
  - Save to database
  - Return final LLM output string

Does NOT:
  - Build prompts          → llm/prompt_builder.py
  - Call LLM directly      → llm/summarizer.py  (via summarize_data / call_llm)
  - Clean text             → utils/cleaner.py
  - Multi-stage pipeline   → llm/pipeline.py
"""

from __future__ import annotations

import asyncio
import os
import json
import logging
from pathlib import Path

# ── Utils ────────────────────────────────────────────────────────────────────
from utils.source_detector import detect_source
from utils.cleaner         import clean_processor_output

# ── LLM layer ────────────────────────────────────────────────────────────────
from llm.summarizer  import summarize_data, call_llm
from llm.pipeline    import classify, extract_guidance, enrich

# ── Database ─────────────────────────────────────────────────────────────────
from database.db import save_resource, init_db

# ── Processors ───────────────────────────────────────────────────────────────
from processors.youtube_processor   import process_youtube
from processors.github_processor    import process_github
from processors.web_processor       import process_web
from processors.instagram_processor import process_instagram
from processors.text_processor      import process_text
from processors.image_processor     import process_image

import time
CURRENT_SESSION_ID = int(time.time())

from dotenv import load_dotenv
load_dotenv()   
import os
print(f"  [DEBUG] INSTAGRAM_USERNAME={os.getenv('INSTAGRAM_USERNAME')}")    

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

_LOCAL_SOURCE_TYPES: set[str] = {
    "local_image", "local_video", "local_audio",
    "pdf_document", "word_document", "spreadsheet",
    "presentation", "ebook", "code_file", "notebook",
    "data_file", "archive", "plain_text_file",
}

_WEB_SOURCE_TYPES: set[str] = {
    "web", "medium_article", "substack_article",
    "notion_page", "arxiv_paper", "reddit_post",
    "reddit_subreddit", "linkedin_post", "linkedin_article",
    "linkedin_company", "huggingface_model", "huggingface_dataset",
    "huggingface_space", "pastebin", "loom_video", "vimeo_video",
    "image_url", "pdf_url", "audio_url", "code_url", "data_url", "video_url",
}

_UNSUPPORTED_SOURCES: dict[str, str] = {
    "linkedin_profile": (
        "LinkedIn profiles are login-protected and cannot be scraped.\n"
        "Tip: Copy the person's bio or experience text and paste it directly."
    ),
    "linkedin_company": (
        "LinkedIn company pages are login-protected and cannot be scraped.\n"
        "Tip: Paste the company description text directly instead."
    ),
    "unsupported": (
        "This URL could not be processed (JavaScript-only or login-required site).\n"
        "Tip: Copy the page text and paste it directly into the chat."
    ),
}

_IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}

_TRUST_RULE_BASED: set[str] = {
    "youtube_video", "youtube_shorts", "youtube_playlist",
    "github_repo", "github_file", "github_gist",
    "local_image", "pdf_document",
    "instagram_post", "instagram_reel",
}


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_json(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)

def _make_vault_title(llm_output: str | None, fallback: str) -> str:
    if llm_output:
        first = llm_output.strip().splitlines()[0].strip()
        if 8 <= len(first) <= 80:
            return first
    return fallback[:80]

def _make_vault_snippet(llm_output: str | None) -> str | None:
    if not llm_output:
        return None
    s = " ".join(llm_output.strip().split())
    return s[:160]


def _save(
    source:       str,
    url:          str | None,
    title:        str,
    raw_input:    dict,
    raw_data:     dict | None = None,
    cleaned_data: dict | None = None,
    llm_output:   str  | None = None,
    status:       str        = "success",
    error:        str  | None = None,
    session_id: int = CURRENT_SESSION_ID,
) -> None:
    try:
        # guarantee raw_input is never empty/None
        if not raw_input:
            raw_input = {"kind": source}

        # Better vault metadata generation
        vault_title = title or url or source

        if raw_data:
            if raw_data.get("repo"):
                vault_title = raw_data["repo"]

            elif raw_data.get("title"):
                vault_title = raw_data["title"]

        vault_snippet = None

        if raw_data:
            if raw_data.get("description"):
                vault_snippet = raw_data["description"]

            elif raw_data.get("overview"):
                vault_snippet = raw_data["overview"][:160]

        # fallback to LLM output if nothing else exists
        if not vault_snippet:
            vault_snippet = _make_vault_snippet(llm_output)

        files = None
        if raw_data and raw_data.get("files"):
            files = raw_data["files"]

        save_resource(
            source        = source,
            url           = url,
            title         = title,
            raw_input     = raw_input,
            raw_data      = raw_data,
            cleaned_data  = cleaned_data,
            llm_output    = llm_output,
            files         = files,
            status        = status,
            error         = error,
            vault_title   = vault_title,
            vault_snippet = vault_snippet,
            session_id    = session_id,
        )
    except Exception as e:
        logger.error(f"DB save failed: {e}")

def _friendly_error(url: str, source: str, err: str) -> str:
    if "empty data" in err:
        return (
            f"⚠️ Could not extract content from this URL.\n\n"
            f"Possible reasons:\n"
            f"  • The site requires JavaScript to render (React/Next.js)\n"
            f"  • The site blocks automated access\n"
            f"  • The page has no readable text content\n\n"
            f"Tip: Copy the page text and paste it directly into the chat."
        )
    if "404" in err:
        return f"⚠️ Page not found (404): {url}"
    if "timeout" in err.lower():
        return (
            f"⚠️ Request timed out for: {url}\n"
            f"Try again in a moment, or paste the content directly."
        )
    return f"⚠️ Error processing link: {err}"


async def _detect_source_smart(url: str) -> str:
    """
    Two-stage source detection:
      Stage 1 — fast regex via detect_source()
      Stage 2 — async LLM classifier fallback for ambiguous results
    """
    rule_based = detect_source(url)

    if rule_based in _TRUST_RULE_BASED:
        return rule_based

    _AMBIGUOUS = {"unknown", "web", "github_user"}
    if rule_based in _AMBIGUOUS:
        print(f"  [ROUTER] Rule-based='{rule_based}', asking LLM classifier...")
        result = await classify(url)                     # ← await
        print(
            f"  [ROUTER] LLM classified='{result['source_type']}' "
            f"(confidence={result['confidence']}, reason={result['reason']})"
        )
        if result["confidence"] in ("high", "medium"):
            return result["source_type"]

    return rule_based


async def _run_pipeline(user_input: str, raw_data: dict, source_type: str) -> str:
    """
    Runs stages 2–4 of the multi-stage pipeline (all now async).
      Stage 2 — extract_guidance()
      Stage 3 — clean + enrich()
      Stage 4 — summarize_data()
    """
    guidance = await extract_guidance(user_input, source_type)   # ← await
    cleaned  = clean_processor_output(raw_data)                  # sync — pure dict ops
    enriched = await enrich(cleaned, guidance)                   # ← await
    return await summarize_data(enriched)                        # ← await


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE URL  (now async)
# ─────────────────────────────────────────────────────────────────────────────

async def process_link(url: str) -> str:
    if not url or not url.strip():
        return "No URL provided."

    url    = url.strip()
    source = await _detect_source_smart(url)                     # ← await

    if source in _LOCAL_SOURCE_TYPES:
        return (
            f"'{url}' looks like a local file path.\n"
            f"Use the file attachment button instead."
        )

    if source in _UNSUPPORTED_SOURCES:
        return f"⚠️ Cannot process this link.\n\n{_UNSUPPORTED_SOURCES[source]}"

    raw_data = None

    try:
        if source.startswith("youtube"):
            raw_data = process_youtube(url)
        elif source.startswith("github"):
            raw_data = process_github(url)
        elif source.startswith("instagram"):
            raw_data = process_instagram(url)
        else:
            raw_data = await process_web(url)

        if not raw_data:
            raise ValueError(f"Processor returned empty data for source '{source}'")

        raw_data["source_type"] = source

        llm_output = await _run_pipeline(url, raw_data, source)  # ← await

        _save(
            source       = source,
            url          = url,
            title        = raw_data.get("title", ""),
            raw_input    = {"url": url},
            raw_data     = raw_data,
            cleaned_data = clean_processor_output(raw_data),
            llm_output   = llm_output,
        )
        return llm_output

    except Exception as e:
        err = str(e)
        logger.error(f"process_link failed for {url}: {err}")
        _save(
            source    = source,
            url       = url,
            title     = "",
            raw_input = {"url": url},
            status    = "error",
            error     = err,
        )
        return _friendly_error(url, source, err)


# ─────────────────────────────────────────────────────────────────────────────
# PLAIN TEXT  (now async)
# ─────────────────────────────────────────────────────────────────────────────

async def process_text_input(text: str) -> str:
    if not text or not text.strip():
        return "No text provided."

    text = text.strip()

    try:
        clf    = await classify(text[:800])                      # ← await
        source = clf.get("source_type", "plain_text")
        print(f"  [ROUTER] Text classified as: {source} ({clf.get('confidence')} confidence)")

        from llm.llm_classifier import is_supported
        if not is_supported(source):
                return f"⚠️ Cannot process this.\n\n{_UNSUPPORTED_SOURCES.get(source, 'This input type is not supported.')}"

        raw_data                = process_text(text, source_type=source)
        raw_data["source_type"] = source

        llm_output = await _run_pipeline(text[:300], raw_data, source)  # ← await

        _save(
            source       = source,
            url          = None,
            title        = raw_data.get("title", text[:80]),
            raw_input    = {"text": text[:300]},
            raw_data     = raw_data,
            cleaned_data = clean_processor_output(raw_data),
            llm_output   = llm_output,
        )
        return llm_output

    except Exception as e:
        logger.error(f"process_text_input failed: {e}")
        return f"Error processing text: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE  (now async)
# ─────────────────────────────────────────────────────────────────────────────

async def process_image_input(image_path: str) -> str:
    image_path = str(Path(image_path).resolve())

    if not os.path.exists(image_path):
        return f"Image not found: {image_path}"

    try:
        raw_data = process_image(image_path)

        ocr_text = raw_data.get("ocr_text", "") or raw_data.get("content", "")
        clf      = await classify(ocr_text[:500])                # ← await
        print(
            f"  [ROUTER] Image OCR classified as: {clf['source_type']} "
            f"({clf['confidence']} confidence)"
        )

        raw_data["source_type"] = "local_image"

        llm_output = await _run_pipeline(ocr_text[:300], raw_data, "local_image")  # ← await

        _save(
            source       = "local_image",
            url          = None,
            title        = raw_data.get("title", Path(image_path).name),
            raw_input    = {
                "kind":       "local_image",
                "image_path": image_path,
                "filename":   Path(image_path).name,
            },
            raw_data     = raw_data,
            cleaned_data = clean_processor_output(raw_data),
            llm_output   = llm_output,
        )
        return llm_output

    except Exception as e:
        logger.error(f"process_image_input failed for {image_path}: {e}")
        return f"Error processing image '{Path(image_path).name}': {e}"


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL FILE  (now async)
# ─────────────────────────────────────────────────────────────────────────────

async def process_local_file(file_path: str) -> str:
    file_path = str(Path(file_path).resolve())

    if not os.path.exists(file_path):
        return f"File not found: {file_path}"

    source = detect_source(file_path)

    if source == "local_image":
        return await process_image_input(file_path)              # ← await

    if source in ("plain_text_file", "code_file", "notebook", "data_file", "pdf_document"):
        try:
            text = Path(file_path).read_text(encoding="utf-8", errors="replace")
            return await process_text_input(text)                # ← await
        except Exception as e:
            return f"Error reading file '{Path(file_path).name}': {e}"

    return (
        f"Unsupported local file type: {source}\n"
        f"File: {Path(file_path).name}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MIXED INPUT  ← primary entry point called by web/app.py FastAPI /chat
# (now async)
# ─────────────────────────────────────────────────────────────────────────────

async def process_input(
    text:        str       = "",
    image_paths: list[str] = None,
) -> str:
    """
    Handles any combination of URLs, plain text, and image file paths.
    Called by the FastAPI /chat endpoint in web/app.py.
    Now fully async — no event loop blocking.
    """
    image_paths = [p for p in (image_paths or []) if p]
    parts: list[str] = []

    lines     = [l.strip() for l in (text or "").splitlines() if l.strip()]
    urls      = [
        l for l in lines
        if (l.startswith("http://") or l.startswith("https://"))
        and not os.path.exists(l)
    ]
    plain     = [l for l in lines if l not in urls]
    plain_str = "\n".join(plain).strip()

    for url in urls:
        try:
            result = await process_link(url)                     # ← await
            parts.append(f"[{url}]\n{result}")
        except Exception as e:
            parts.append(f"[{url}]\nError: {e}")

    if plain_str:
        try:
            result = await process_text_input(plain_str)         # ← await
            parts.append(f"[Text Note]\n{result}")
        except Exception as e:
            parts.append(f"[Text Note]\nError: {e}")

    for img_path in image_paths:
        fname = Path(img_path).name
        try:
            if not os.path.exists(img_path):
                parts.append(f"[Image: {fname}]\nError: file not found.")
                continue
            result = await process_image_input(img_path)         # ← await
            parts.append(f"[Image: {fname}]\n{result}")
        except Exception as e:
            parts.append(f"[Image: {fname}]\nError: {e}")

    if not parts:
        return "Nothing to process. Please provide a link, some text, or an image."

    if len(parts) == 1:
        return parts[0].split("\n", 1)[-1].strip()

    combined_prompt = (
        "The user provided multiple resources. "
        "Summarise each one briefly, then give a combined insight:\n\n"
        + "\n\n---\n\n".join(parts)
    )
    try:
        return await call_llm(combined_prompt)                   # ← await
    except Exception:
        return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY  (python main.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    init_db()

    async def _cli_loop() -> None:
        print("AI Resource Agent")
        print("─" * 40)
        print("Paste a URL, type some text, or enter a local file/image path.")
        print("Type 'exit' to quit.\n")

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

            p = Path(user_input)
            if p.exists() and p.suffix.lower() in _IMAGE_EXTS:
                result = await process_image_input(user_input)   # ← await
            elif p.exists() and p.is_file():
                result = await process_local_file(user_input)    # ← await
            else:
                result = await process_input(text=user_input)    # ← await

            print(f"\n{result}\n")

    asyncio.run(_cli_loop())
