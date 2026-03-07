import re
import os
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
# Whisper model cache — loaded once, reused across calls
_whisper_model = None

def _get_whisper_model(size: str = "base"):
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print(f"[youtube_processor] Loading Whisper '{size}' model...")
        _whisper_model = whisper.load_model(size)
    return _whisper_model

# ─────────────────────────────────────────────────────────────────────────────
# VIDEO ID EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def get_video_id(url: str) -> str | None:
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    if "shorts/" in url:
        return url.split("shorts/")[1].split("?")[0]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# TEXT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s,.!?'-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_meaningful(text: str) -> bool:
    cleaned = re.sub(r"[^\w\s]", "", text).strip()
    return len(cleaned.split()) >= 4


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — METADATA ONLY (no comments — fast and reliable)
# ─────────────────────────────────────────────────────────────────────────────

def get_youtube_metadata(url: str) -> dict:
    """
    Fetches title, description, duration, view count, channel.
    Comments are intentionally excluded here — they are fetched
    separately in get_youtube_comments() so a comment failure
    never blocks metadata.
    """
    ydl_opts = {
        "quiet":         True,
        "skip_download": True,
        "extract_flat":  False,
        "getcomments":   False,       # ← KEY FIX: no comments here
        "no_warnings":   False,       # keep warnings visible for debugging
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        if not info:
            return {}

        return {
            "title":       info.get("title",        ""),
            "description": info.get("description",  ""),
            "duration":    info.get("duration",      0),     # seconds
            "view_count":  info.get("view_count",    0),
            "channel":     info.get("channel",       ""),
            "upload_date": info.get("upload_date",   ""),    # YYYYMMDD
            "tags":        info.get("tags",          []),
        }

    except yt_dlp.utils.DownloadError as e:
        print(f"[youtube_processor] yt-dlp download error: {e}")
        return {}
    except Exception as e:
        print(f"[youtube_processor] Metadata fetch failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — COMMENTS (separate call — failure is non-fatal)
# ─────────────────────────────────────────────────────────────────────────────

def get_youtube_comments(url: str, max_fetch: int = 30) -> list[str]:
    """
    Separate comment fetch — isolated so failures don't affect metadata.
    Returns empty list on any error (non-fatal).
    """
    ydl_opts = {
        "quiet":         True,
        "skip_download": True,
        "extract_flat":  False,
        "getcomments":   True,
        "no_warnings":   True,        # suppress "Incomplete data" warnings here
        "extractor_args": {
            "youtube": {
                "comment_sort": ["top"],
                "max_comments": [str(max_fetch)],
            }
        },
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        if not info:
            return []

        raw = []
        for c in (info.get("comments") or []):
            text = (c.get("text") or "").strip()
            if text:
                raw.append(text)
        return raw

    except Exception as e:
        # Comment fetch failure is not an error — just skip comments
        print(f"[youtube_processor] Comments unavailable (non-fatal): {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — TRANSCRIPT
# ─────────────────────────────────────────────────────────────────────────────

def _transcribe_with_whisper(url: str, source_type: str = "youtube_shorts") -> str:
    """
    Downloads video, extracts audio, transcribes with Whisper.
    Only runs for youtube_shorts and instagram_reel — not full videos.
    Cleans up temp files after transcription.
    """
    import tempfile

    # Safety check — only run for short-form content
    if source_type not in ("youtube_shorts", "instagram_reel"):
        return ""

    tmp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(tmp_dir, "audio.mp3")

    try:
        # Step 1 — Download audio only via yt-dlp
        print(f"[youtube_processor] Whisper: downloading audio for transcription...")
        ydl_opts = {
            "quiet":       True,
            "format":      "bestaudio/best",
            "outtmpl":     os.path.join(tmp_dir, "audio.%(ext)s"),
            "postprocessors": [{
                "key":            "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",   # low quality = faster + smaller
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Find the downloaded audio file
        audio_files = [
            f for f in os.listdir(tmp_dir)
            if f.endswith(".mp3")
        ]
        if not audio_files:
            print("[youtube_processor] Whisper: no audio file found after download")
            return ""

        # Safety — skip if audio file is too large (> 25MB = ~30 min video)
        audio_path = os.path.join(tmp_dir, audio_files[0])
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if file_size_mb > 25:
            print(f"[youtube_processor] Whisper: audio too large ({file_size_mb:.1f}MB), skipping")
            return ""
        print(f"[youtube_processor] Whisper: audio size {file_size_mb:.1f}MB")
        print(f"[youtube_processor] Whisper: transcribing {audio_path}...")

        # Step 2 — Transcribe with Whisper
        model = _get_whisper_model("base")   # base = fast, good enough for shorts
        result = model.transcribe(audio_path, fp16=False)
        text   = result.get("text", "").strip()

        if text:
            print(f"[youtube_processor] Whisper: {len(text.split())} words transcribed")
        return text

    except ImportError:
        print("[youtube_processor] Whisper not installed. Run: pip install openai-whisper")
        return ""
    except Exception as e:
        print(f"[youtube_processor] Whisper transcription failed: {e}")
        return ""
    finally:
        # Step 3 — Always clean up temp files
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def get_transcript(url: str) -> str:
    """
    Priority order:
      1. Manual English/Hindi captions    (youtube_transcript_api)
      2. Auto-generated English/Hindi     (youtube_transcript_api)
      3. Any language translated to EN    (youtube_transcript_api)
      4. pytubefix captions fallback      (when transcript_api blocked)
    Returns empty string if nothing available.
    """
    video_id = get_video_id(url)
    if not video_id:
        return ""

    # ── Attempts 1–3: youtube_transcript_api ─────────────────────────────
    try:
        data = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en", "hi"]
        )
        return " ".join(t["text"] for t in data).strip()
    except Exception:
        pass

    try:
        transcripts    = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript_obj = transcripts.find_generated_transcript(["en", "hi"])
        data           = transcript_obj.fetch()
        return " ".join(t["text"] for t in data).strip()
    except Exception:
        pass

    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        for t in transcripts:
            try:
                data = t.translate("en").fetch()
                return " ".join(item["text"] for item in data).strip()
            except Exception:
                continue
    except Exception:
        pass

    # ── Attempt 4: pytubefix fallback ─────────────────────────────────────
    print(f"[youtube_processor] transcript_api failed — trying pytubefix fallback")
    try:
        from pytubefix import YouTube

        yt      = YouTube(url)
        caption = (
            yt.captions.get("en")
            or yt.captions.get("a.en")   # auto-generated English
            or yt.captions.get("en-US")
            or (list(yt.captions.values())[0] if yt.captions else None)
        )

        if caption:
            try:
                text = caption.generate_srt_captions()

                # Guard — sometimes returns the object itself, not a string
                if not isinstance(text, str):
                    raise ValueError(f"generate_srt_captions returned non-string: {type(text)}")

                # Strip SRT timestamps — keep only text lines
                lines = []
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.isdigit():
                        continue
                    if "-->" in line:
                        continue
                    lines.append(line)

                result = " ".join(lines).strip()
                if result:
                    print(f"[youtube_processor] pytubefix transcript: {len(result.split())} words")
                    return result

            except Exception as cap_err:
                print(f"[youtube_processor] Caption extraction failed: {cap_err}")

                # Try other available captions before giving up
                try:
                    yt = YouTube(url)
                    for code, cap in yt.captions.items():
                        if code == caption.code:
                            continue   # skip the one that failed
                        try:
                            text = cap.generate_srt_captions()
                            if not isinstance(text, str):
                                continue
                            lines = [
                                l.strip() for l in text.splitlines()
                                if l.strip()
                                and not l.strip().isdigit()
                                and "-->" not in l
                            ]
                            result = " ".join(lines).strip()
                            if result:
                                print(f"[youtube_processor] pytubefix fallback caption [{code}]: {len(result.split())} words")
                                return result
                        except Exception:
                            continue
                except Exception:
                    pass


    except ImportError:
        print("[youtube_processor] pytubefix not installed. Run: pip install pytubefix")
    except Exception as e:
        print(f"[youtube_processor] pytubefix fallback failed: {e}")

    # ── Attempt 5: Whisper (shorts + reels only — downloads audio) ────────
    # Only trigger for short-form content — too slow for full videos
    _short_form = ("shorts/", "instagram.com/reel")
    if any(s in url for s in _short_form):
        print(f"[youtube_processor] All caption methods failed — trying Whisper")
        whisper_text = _transcribe_with_whisper(
            url,
            source_type="youtube_shorts" if "shorts/" in url else "instagram_reel"
        )
        if whisper_text:
            return whisper_text

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# COMMENT FILTER
# ─────────────────────────────────────────────────────────────────────────────

def get_top_comments(raw_comments: list, max_count: int = 12) -> list:
    seen   = set()
    result = []

    for comment in raw_comments:
        cleaned = clean_text(comment)
        key     = cleaned.lower()

        if not cleaned:
            continue
        if key in seen:
            continue
        if not is_meaningful(cleaned):
            continue

        seen.add(key)
        result.append(cleaned)

        if len(result) >= max_count:
            break

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

def process_youtube(url: str) -> dict:
    """
    3-stage fetch — each stage is independent.
    A failure in comments or transcript never blocks the metadata.

    Stage 1: Metadata  (title, description, views, channel) — fast, reliable
    Stage 2: Comments  (top 12 meaningful comments)         — optional, non-fatal
    Stage 3: Transcript (manual → auto → translated)        — best-effort
    """
    # Stage 1 — always run, required
    metadata     = get_youtube_metadata(url)

    if not metadata:
        print(f"[youtube_processor] Metadata fetch failed entirely for: {url}")
        return {}

    # Stage 2 — optional, never crashes the pipeline
    raw_comments = get_youtube_comments(url)
    top_comments = get_top_comments(raw_comments, max_count=12)

    # Stage 3 — best effort
    transcript   = get_transcript(url)

    if transcript:
        print(f"[youtube_processor] Transcript: {len(transcript.split())} words")
    else:
        print(f"[youtube_processor] No transcript available — using description only")

    return {
        "title":        metadata.get("title",       ""),
        "description":  clean_text(metadata.get("description", "")),
        "channel":      metadata.get("channel",     ""),
        "view_count":   metadata.get("view_count",  0),
        "duration":     metadata.get("duration",    0),
        "upload_date":  metadata.get("upload_date", ""),
        "tags":         metadata.get("tags",        []),
        "transcript":   transcript,
        "top_comments": top_comments,
    }

