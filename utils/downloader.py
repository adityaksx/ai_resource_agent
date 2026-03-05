"""
downloader.py
-------------
Downloads content from all supported sources:

  URLs:
    - YouTube Shorts only  (full videos/playlists/channels/live → transcript-only)
    - Instagram (reel, post, story)
    - Twitter / X          (short clips)
    - GitHub               (repo clone, single file, gist)
    - ArXiv                (PDF download)
    - Direct file URLs     (image, pdf, zip, mp3, …)
    - Web articles         (readable text via trafilatura + BS4 fallback)

  Skipped (no download, use transcript fallback):
    - YouTube video, playlist, channel, live
    - TikTok, Loom, Vimeo, Reddit video
    - Instagram profile (bulk)

  Returns a structured DownloadResult dataclass with:
    - saved file/dir path(s)
    - source type
    - metadata (title, duration, size, etc.)
    - skipped flag + reason
    - error if failed
"""

from __future__ import annotations

import os
import re
import json
import logging
import hashlib
import requests
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# STORAGE DIRS
# ─────────────────────────────────────────────

BASE_DIR  = Path("storage")
RAW_DIR   = BASE_DIR / "raw"
VIDEO_DIR = BASE_DIR / "videos"
IMAGE_DIR = BASE_DIR / "images"
AUDIO_DIR = BASE_DIR / "audio"
REPO_DIR  = BASE_DIR / "repos"
PDF_DIR   = BASE_DIR / "pdfs"
DATA_DIR  = BASE_DIR / "data"

ALL_DIRS = [RAW_DIR, VIDEO_DIR, IMAGE_DIR, AUDIO_DIR, REPO_DIR, PDF_DIR, DATA_DIR]

def _ensure_dirs():
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────

@dataclass
class DownloadResult:
    source_type: str
    url:         str
    paths:       list[str] = field(default_factory=list)
    metadata:    dict      = field(default_factory=dict)
    error:       Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.paths)

    @property
    def skipped(self) -> bool:
        return bool(self.metadata.get("skipped"))

    @property
    def skip_reason(self) -> Optional[str]:
        return self.metadata.get("reason")

    @property
    def primary_path(self) -> Optional[str]:
        return self.paths[0] if self.paths else None

    def __str__(self):
        if self.skipped:
            return f"Skipped({self.source_type}): {self.skip_reason}"
        if self.error:
            return f"DownloadError({self.source_type}): {self.error}"
        return f"Downloaded({self.source_type}): {', '.join(self.paths)}"


# ─────────────────────────────────────────────
# SKIP HELPER
# ─────────────────────────────────────────────

def _skip(url: str, source_type: str, reason: str) -> DownloadResult:
    """
    Return a no-download result. Caller should fall back to transcript/text extraction.
    This is NOT an error — it is intentional.
    """
    logger.info(f"Skipping download for '{source_type}': {reason}")
    return DownloadResult(
        source_type=source_type,
        url=url,
        paths=[],
        metadata={"skipped": True, "reason": reason},
    )


# ─────────────────────────────────────────────
# YT-DLP HELPER
# ─────────────────────────────────────────────

def _ytdlp_download(
    url: str,
    output_dir: Path,
    source_type: str,
    audio_only: bool = False,
    max_resolution: int = 1080,
    extra_opts: list[str] | None = None,
) -> DownloadResult:
    """Generic yt-dlp downloader with metadata extraction."""
    _ensure_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)

    template = str(output_dir / "%(title).80s_%(id)s.%(ext)s")

    cmd = ["yt-dlp", "--no-playlist"]

    if audio_only:
        cmd += [
            "-x",
            "--audio-format", "mp3",
            "--audio-quality", "192K",
        ]
    else:
        cmd += [
            "-S", f"res:{max_resolution},ext:mp4:m4a",
            "--merge-output-format", "mp4",
        ]

    cmd += [
        "--write-info-json",
        "--no-write-thumbnail",
        "-o", template,
    ]

    if extra_opts:
        cmd += extra_opts

    cmd.append(url)

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        err = result.stderr[-400:] if result.stderr else "Unknown yt-dlp error"
        return DownloadResult(source_type=source_type, url=url, error=err)

    # Collect saved files (exclude info json)
    saved = [
        str(p) for p in output_dir.glob("*")
        if p.is_file() and p.suffix != ".json" and not p.name.startswith(".")
    ]

    # Parse metadata from info json
    metadata = {}
    for p in output_dir.glob("*.info.json"):
        try:
            info = json.loads(p.read_text(encoding="utf-8"))
            metadata = {
                "title":       info.get("title", ""),
                "duration":    info.get("duration"),
                "uploader":    info.get("uploader", ""),
                "view_count":  info.get("view_count"),
                "upload_date": info.get("upload_date", ""),
                "description": (info.get("description") or "")[:500],
            }
            p.unlink()
        except Exception:
            pass

    return DownloadResult(
        source_type=source_type,
        url=url,
        paths=sorted(saved),
        metadata=metadata,
    )


# ─────────────────────────────────────────────
# YOUTUBE — SHORTS ONLY
# ─────────────────────────────────────────────

def download_youtube_shorts(url: str) -> DownloadResult:
    """
    Download YouTube Shorts only at max 480p to save storage.
    Full videos, playlists, channels, and live streams are skipped.
    """
    _ensure_dirs()
    return _ytdlp_download(
        url, VIDEO_DIR,
        source_type="youtube_shorts",
        audio_only=False,
        max_resolution=480,
    )


# ─────────────────────────────────────────────
# INSTAGRAM — REELS, POSTS, STORIES
# ─────────────────────────────────────────────

def download_instagram(url: str) -> DownloadResult:
    """Download Instagram reel, post, or story via yt-dlp."""
    source_type = (
        "instagram_reel"  if "/reel/"    in url else
        "instagram_post"  if "/p/"       in url else
        "instagram_story" if "/stories/" in url else
        "instagram_post"
    )
    return _ytdlp_download(url, VIDEO_DIR, source_type=source_type)


# ─────────────────────────────────────────────
# TWITTER / X  — short clips only
# ─────────────────────────────────────────────

def download_twitter(url: str) -> DownloadResult:
    """Download Twitter/X embedded video clips (usually short)."""
    return _ytdlp_download(url, VIDEO_DIR, source_type="twitter_tweet")


# ─────────────────────────────────────────────
# GITHUB — repo, file, gist
# ─────────────────────────────────────────────

def download_github(url: str) -> DownloadResult:
    """Clone a repo (shallow), download a single file, or fetch a gist."""
    _ensure_dirs()

    if "gist.github.com" in url:
        return _clone_repo(url, REPO_DIR, "github_gist")

    parts = [p for p in urlparse(url).path.split("/") if p]

    # Single file: github.com/user/repo/blob/branch/path/to/file
    if len(parts) >= 4 and parts[2] in ("blob", "raw"):
        raw_url   = url.replace("/blob/", "/raw/")
        filename  = parts[-1]
        save_path = RAW_DIR / filename
        return _download_direct_file(raw_url, save_path, "github_file")

    return _clone_repo(url, REPO_DIR, "github_repo")


def _clone_repo(url: str, dest_dir: Path, source_type: str) -> DownloadResult:
    repo_name = url.rstrip("/").split("/")[-1].removesuffix(".git")
    path = dest_dir / repo_name

    if path.exists():
        try:
            import git
            git.Repo(str(path)).remotes.origin.pull()
            return DownloadResult(
                source_type=source_type, url=url,
                paths=[str(path)],
                metadata={"status": "pulled_latest", "repo": repo_name},
            )
        except Exception:
            return DownloadResult(
                source_type=source_type, url=url,
                paths=[str(path)],
                metadata={"status": "already_exists"},
            )

    try:
        import git
        git.Repo.clone_from(url, str(path), depth=1)
        return DownloadResult(
            source_type=source_type, url=url,
            paths=[str(path)],
            metadata={"repo": repo_name},
        )
    except Exception as e:
        return DownloadResult(source_type=source_type, url=url, error=str(e))


# ─────────────────────────────────────────────
# ARXIV — PDF
# ─────────────────────────────────────────────

def download_arxiv(url: str) -> DownloadResult:
    """Download ArXiv paper as PDF."""
    _ensure_dirs()

    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^\s/?#]+)", url)
    if not match:
        return DownloadResult(source_type="arxiv_paper", url=url,
                              error="Could not parse ArXiv ID from URL")

    arxiv_id  = match.group(1).removesuffix(".pdf")
    pdf_url   = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    safe_id   = arxiv_id.replace("/", "_")
    save_path = PDF_DIR / f"arxiv_{safe_id}.pdf"

    return _download_direct_file(
        pdf_url, save_path, "arxiv_paper",
        metadata={"arxiv_id": arxiv_id}
    )


# ─────────────────────────────────────────────
# DIRECT FILE URL  (.pdf, .jpg, .zip, .mp3, …)
# ─────────────────────────────────────────────

def _download_direct_file(
    url: str,
    save_path: Path,
    source_type: str,
    chunk_size: int = 65536,
    metadata: dict | None = None,
) -> DownloadResult:
    """Stream-download any direct file URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ai-resource-agent/1.0)"}
        with requests.get(url, headers=headers, stream=True, timeout=30) as r:
            r.raise_for_status()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)

        size = save_path.stat().st_size
        meta = {"size_bytes": size, "size_kb": round(size / 1024, 1)}
        if metadata:
            meta.update(metadata)

        return DownloadResult(
            source_type=source_type, url=url,
            paths=[str(save_path)], metadata=meta,
        )
    except Exception as e:
        return DownloadResult(source_type=source_type, url=url, error=str(e))


def download_direct_file(url: str) -> DownloadResult:
    """Route a direct file URL to the correct storage directory."""
    _ensure_dirs()
    parsed   = urlparse(url)
    ext      = Path(parsed.path).suffix.lower()
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    filename = Path(parsed.path).name or f"file_{url_hash}{ext}"

    ext_dir_map = {
        ".pdf":  PDF_DIR,
        ".jpg":  IMAGE_DIR, ".jpeg": IMAGE_DIR, ".png":  IMAGE_DIR,
        ".gif":  IMAGE_DIR, ".webp": IMAGE_DIR, ".svg":  IMAGE_DIR,
        ".bmp":  IMAGE_DIR, ".tiff": IMAGE_DIR,
        ".mp3":  AUDIO_DIR, ".wav":  AUDIO_DIR, ".ogg":  AUDIO_DIR,
        ".flac": AUDIO_DIR, ".m4a":  AUDIO_DIR,
        ".mp4":  VIDEO_DIR, ".mov":  VIDEO_DIR,
        ".zip":  DATA_DIR,  ".csv":  DATA_DIR,  ".json": DATA_DIR,
        ".xlsx": DATA_DIR,  ".xml":  DATA_DIR,
    }
    out_dir   = ext_dir_map.get(ext, RAW_DIR)
    save_path = out_dir / filename

    return _download_direct_file(url, save_path, "direct_file")


# ─────────────────────────────────────────────
# WEB ARTICLE
# ─────────────────────────────────────────────

def download_webpage(url: str) -> DownloadResult:
    """
    Extract clean readable text from any webpage.
    Uses trafilatura (primary) with BeautifulSoup fallback.
    """
    _ensure_dirs()

    headers = {"User-Agent": "Mozilla/5.0 (compatible; ai-resource-agent/1.0)"}
    text  = ""
    title = ""

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        html = response.text

        # Primary: trafilatura strips ads, nav, boilerplate
        try:
            import trafilatura
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            )
            if extracted:
                text = extracted
        except ImportError:
            pass

        # Fallback: BeautifulSoup
        if not text:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            paras = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
            text  = "\n\n".join(p for p in paras if len(p) > 40)

    except Exception as e:
        return DownloadResult(source_type="web", url=url, error=str(e))

    if not text:
        return DownloadResult(source_type="web", url=url,
                              error="No readable content extracted")

    domain    = urlparse(url).netloc.replace(".", "_").replace("www_", "")
    url_hash  = hashlib.md5(url.encode()).hexdigest()[:6]
    save_path = RAW_DIR / f"{domain}_{url_hash}.txt"
    save_path.write_text(text, encoding="utf-8")

    return DownloadResult(
        source_type="web", url=url,
        paths=[str(save_path)],
        metadata={"title": title, "chars": len(text), "words": len(text.split())},
    )


# ─────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────

def download(url: str, source_type: str) -> DownloadResult:
    """
    Route a URL to the correct downloader based on source_type
    (as returned by source_detector.detect_source).

    Skipped sources return a DownloadResult with .skipped == True.
    Caller should fall back to transcript.get_transcript(url) for those.
    """
    url = url.strip()

    # ── YouTube ──────────────────────────────────────────────────────
    if source_type == "youtube_shorts":
        return download_youtube_shorts(url)

    if source_type in ("youtube_video", "youtube_playlist",
                       "youtube_channel", "youtube_live"):
        return _skip(url, source_type, "download_disabled_use_transcript")

    # ── Instagram ────────────────────────────────────────────────────
    if source_type in ("instagram_reel", "instagram_post", "instagram_story"):
        return download_instagram(url)

    if source_type in ("instagram_profile", "instagram_tv"):
        return _skip(url, source_type, "profile_bulk_download_disabled")

    # ── Twitter / X ──────────────────────────────────────────────────
    if source_type in ("twitter_tweet", "twitter_profile"):
        return download_twitter(url)

    # ── Other video platforms → skip, too large ───────────────────────
    if source_type in ("loom_video", "vimeo_video", "video_url"):
        return _skip(url, source_type, "video_platform_download_disabled_use_transcript")

    # ── GitHub ────────────────────────────────────────────────────────
    if source_type in ("github_repo", "github_profile",
                       "github_file", "github_gist"):
        return download_github(url)

    # ── ArXiv ─────────────────────────────────────────────────────────
    if source_type == "arxiv_paper":
        return download_arxiv(url)

    # ── Direct file URLs ──────────────────────────────────────────────
    if source_type in ("image_url", "pdf_url", "audio_url",
                       "code_url", "data_url", "direct_file"):
        return download_direct_file(url)

    # ── Web / article / text ──────────────────────────────────────────
    if source_type in (
        "web", "medium_article", "substack_article", "notion_page",
        "reddit_post", "reddit_subreddit", "linkedin_post",
        "linkedin_article", "linkedin_company", "huggingface_model",
        "huggingface_dataset", "huggingface_space", "pastebin",
    ):
        return download_webpage(url)

    # ── Unknown — try webpage as last resort ──────────────────────────
    logger.warning(f"Unrecognised source_type '{source_type}' for {url}, trying webpage")
    return download_webpage(url)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.source_detector import detect_source

    parser = argparse.ArgumentParser(description="Download any supported resource")
    parser.add_argument("url",          help="URL to download")
    parser.add_argument("--type",       default=None, help="Override source type detection")
    parser.add_argument("--audio-only", action="store_true",
                        help="YouTube Shorts: extract audio only (mp3)")
    args = parser.parse_args()

    source_type = args.type or detect_source(args.url)
    print(f"Source type : {source_type}")

    if source_type == "youtube_shorts" and args.audio_only:
        result = _ytdlp_download(
            args.url, AUDIO_DIR,
            source_type="youtube_shorts",
            audio_only=True,
        )
    else:
        result = download(args.url, source_type)

    if result.skipped:
        print(f"⏭  Skipped  — reason: {result.skip_reason}")
        print("   → Use transcript.get_transcript(url) to extract content instead.")
    elif result.success:
        print(f"✓  Saved to : {', '.join(result.paths)}")
        if result.metadata:
            for k, v in result.metadata.items():
                if v:
                    print(f"   {k}: {v}")
    else:
        print(f"✗  Error    : {result.error}")
