import os
import re
import yt_dlp
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Session file — saves login so you don't re-login on every restart
_SESSION_FILE = str(Path(__file__).parent.parent / "storage" / "instagram_session.json")
_client       = None   # shared instagrapi client instance


# ─────────────────────────────────────────────────────────────────────────────
# TEXT HELPERS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s,.!?'@#-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_meaningful(text: str) -> bool:
    return len(text.split()) >= 4


def get_top_comments(raw_comments: list, max_count: int = 10) -> list:
    seen   = set()
    result = []
    for comment in raw_comments:
        cleaned = clean_text(comment)
        key     = cleaned.lower().strip()
        if not cleaned or key in seen:
            continue
        if not is_meaningful(cleaned):
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= max_count:
            break
    return result


# ─────────────────────────────────────────────────────────────────────────────
# INSTAGRAPI CLIENT  (for profile fetching)
# ─────────────────────────────────────────────────────────────────────────────

def _get_client():
    """
    Returns a logged-in instagrapi Client.
    Reuses the same session across all calls.
    Loads saved session from disk to avoid re-login on every restart.
    """
    global _client

    if _client is not None:
        return _client

    try:
        from instagrapi import Client
        from instagrapi.exceptions import LoginRequired, BadPassword, TwoFactorRequired

        cl = Client()

        username = os.getenv("INSTAGRAM_USERNAME", "").strip()
        password = os.getenv("INSTAGRAM_PASSWORD", "").strip()

        if not username or not password:
            print(
                "[instagram_processor] No credentials found.\n"
                "Add INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD to your .env file."
            )
            return None

        # Try loading saved session first
        if os.path.exists(_SESSION_FILE):
            try:
                cl.load_settings(_SESSION_FILE)
                cl.login(username, password)   # refreshes session token
                print("[instagram_processor] Session loaded from disk.")
                _client = cl
                return _client
            except Exception:
                print("[instagram_processor] Saved session expired, re-logging in...")
                os.remove(_SESSION_FILE)

        # Fresh login
        cl.login(username, password)
        Path(_SESSION_FILE).parent.mkdir(parents=True, exist_ok=True)
        cl.dump_settings(_SESSION_FILE)   # save for next restart
        print("[instagram_processor] Logged in and session saved.")
        _client = cl
        return _client

    except ImportError:
        print(
            "[instagram_processor] instagrapi not installed.\n"
            "Run: pip install instagrapi"
        )
        return None
    except Exception as e:
        print(f"[instagram_processor] Login failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE PROCESSOR  (new — uses instagrapi)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_username(url: str) -> str | None:
    """Extract username from profile URL."""
    # handles: instagram.com/username/ and instagram.com/username
    url = url.rstrip("/")
    parts = url.split("instagram.com/")
    if len(parts) < 2:
        return None
    username = parts[1].split("/")[0].strip()
    return username if username else None


def process_instagram_profile(url: str) -> dict | None:
    """
    Fetches full profile data using instagrapi + your test account.
    Returns bio, follower count, post count, recent post captions.
    """
    username = _extract_username(url)
    if not username:
        print(f"[instagram_processor] Could not extract username from: {url}")
        return None

    cl = _get_client()
    if not cl:
        return None

    try:
        user = cl.user_info_by_username(username)

        # Fetch up to 6 recent post captions for context
        recent_captions = []
        try:
            user_id = cl.user_id_from_username(username)
            medias  = cl.user_medias(user_id, amount=3)
            for m in medias:
                caption = clean_text(m.caption_text or "")
                if caption and is_meaningful(caption):
                    recent_captions.append(caption)
        except Exception:
            pass   # recent posts are optional

        return {
            "source_type":      "instagram_profile",
            "url":              url,
            "username":         user.username,
            "full_name":        user.full_name       or "",
            "bio":              clean_text(user.biography or "")[:300],
            "followers":        user.follower_count,
            "following":        user.following_count,
            "posts":            user.media_count,
            "is_verified":      user.is_verified,
            "is_business":      user.is_business,
            "category":         getattr(user, "category", "") or "",
            "recent_captions":  recent_captions[:3],
        }

    except Exception as e:
        print(f"[instagram_processor] Profile fetch failed for @{username}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# POST / REEL PROCESSOR  (existing — uses yt-dlp, unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────

def get_instagram_metadata(url: str) -> dict | None:
    """
    Fetches post/reel metadata via yt-dlp.
    Comments separated to avoid poisoning metadata fetch.
    """
    ydl_opts = {
        "quiet":         True,
        "no_warnings":   True,
        "skip_download": True,
        "extract_flat":  False,
        "getcomments":   False,          # ← separated, same fix as youtube_processor
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        if not info:
            return None

        return {
            "title":    info.get("title")       or "",
            "caption":  info.get("description") or "",
            "uploader": info.get("uploader")    or "",
            "likes":    info.get("like_count"),
        }

    except yt_dlp.utils.DownloadError as e:
        print(f"[instagram_processor] yt-dlp download error: {e}")
        return None
    except Exception as e:
        print(f"[instagram_processor] Unexpected error: {e}")
        return None


def get_instagram_comments(url: str) -> list[str]:
    """Isolated comment fetch — failure is non-fatal."""
    ydl_opts = {
        "quiet":         True,
        "no_warnings":   True,
        "skip_download": True,
        "extract_flat":  False,
        "getcomments":   True,
        "extractor_args": {
            "instagram": {"max_comments": ["30"]}
        },
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        if not info:
            return []
        return [
            (c.get("text") or "").strip()
            for c in (info.get("comments") or [])
            if (c.get("text") or "").strip()
        ]
    except Exception as e:
        print(f"[instagram_processor] Comments unavailable (non-fatal): {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSOR  (routes by source type)
# ─────────────────────────────────────────────────────────────────────────────

def process_instagram(url: str) -> dict | None:
    """
    Routes Instagram URLs to the correct processor:
      - Profile URLs  (instagram.com/username/)     → instagrapi
      - Post URLs     (instagram.com/p/...)          → yt-dlp
      - Reel URLs     (instagram.com/reel/...)       → yt-dlp
    """
    # ── Profile URL detection ─────────────────────────────────────────────
    url_clean = url.rstrip("/")
    path      = url_clean.split("instagram.com/")[-1]
    segments  = [s for s in path.split("/") if s]

    is_profile = (
        len(segments) == 1
        and segments[0] not in ("p", "reel", "tv", "stories", "explore")
        and not segments[0].startswith("?")
    )

    if is_profile:
        print(f"[instagram_processor] Detected profile URL → using instagrapi")
        return process_instagram_profile(url)

    # ── Post / Reel URL ───────────────────────────────────────────────────
    print(f"[instagram_processor] Detected post/reel URL → using yt-dlp")
    data = get_instagram_metadata(url)

    if not data:
        return None

    raw_comments = get_instagram_comments(url)
    top_comments = get_top_comments(raw_comments, max_count=10)

    is_reel = "/reel/" in url
    return {
        "source_type": "instagram_reel" if is_reel else "instagram_post",
        "url":         url,
        "title":       data.get("title",    ""),
        "caption":     clean_text(data.get("caption",   "")),
        "top_comments": top_comments,
    }
