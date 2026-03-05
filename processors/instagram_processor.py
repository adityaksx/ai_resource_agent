import re
import yt_dlp


# -------------------------
# Clean a single comment
# -------------------------

def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"http\S+", "", text)           # remove URLs
    text = re.sub(r"[^\w\s,.!?'@#-]", "", text)  # remove emojis/junk, keep punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_meaningful(text: str) -> bool:
    """Filter out very short, emoji-only, or spammy comments."""
    words = text.split()
    return len(words) >= 4   # at least 4 real words


# -------------------------
# Get top meaningful comments
# -------------------------

def get_top_comments(raw_comments: list, max_count: int = 10) -> list:
    seen = set()
    result = []

    for comment in raw_comments:
        cleaned = clean_text(comment)
        key = cleaned.lower().strip()

        if not cleaned or key in seen:
            continue
        if not is_meaningful(cleaned):
            continue

        seen.add(key)
        result.append(cleaned)

        if len(result) >= max_count:
            break

    return result


# -------------------------
# yt-dlp metadata extraction
# -------------------------

def get_instagram_metadata(url: str) -> dict | None:
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
        "getcomments": True,
        "extractor_args": {
            "instagram": {
                "max_comments": ["30"],   # fetch 30, filter to 10
            }
        },
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        if not info:
            print(f"[instagram_processor] No info returned for: {url}")
            return None

        raw_comments = []
        for c in (info.get("comments") or []):
            text = (c.get("text") or "").strip()
            if text:
                raw_comments.append(text)

        return {
            "title": info.get("title") or "",
            "caption": info.get("description") or "",
            "uploader": info.get("uploader") or "",
            "likes": info.get("like_count"),
            "raw_comments": raw_comments,
        }

    except yt_dlp.utils.DownloadError as e:
        print(f"[instagram_processor] yt-dlp download error: {e}")
        return None
    except Exception as e:
        print(f"[instagram_processor] Unexpected error: {e}")
        return None


# -------------------------
# Main processor
# -------------------------

def process_instagram(url: str) -> dict | None:
    """
    Extracts caption, uploader, and top meaningful comments
    from an Instagram post, reel, or story URL.
    """
    data = get_instagram_metadata(url)

    if not data:
        return None

    caption = clean_text(data.get("caption", ""))
    top_comments = get_top_comments(data.get("raw_comments", []), max_count=10)

    return {
        "url": url,
        "title": data.get("title", ""),
        "caption": caption,
        "uploader": data.get("uploader", ""),
        "likes": data.get("likes"),
        "top_comments": top_comments,    # 10 meaningful, unique, clean comments
    }
