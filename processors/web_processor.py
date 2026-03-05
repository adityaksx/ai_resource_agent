import re
import urllib3
import requests
import trafilatura
from trafilatura.settings import use_config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

MAX_CONTENT_CHARS = 8000

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

GOOGLEBOT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
}

JUNK_SIGNALS = [
    "google search troubleshooting",
    "feedback submission",
    "sign in to continue",
    "subscribe to read",
    "create an account",
    "log in to access",
    "access denied",
    "enable javascript",
    "please verify you are a human",
    "captcha",
    "403 forbidden",
    "404 not found",
    "just a moment",
    "checking your browser",
    "you've reached your",
    "subscribe for full access",
]


# -------------------------
# Trafilatura config
# -------------------------

def get_trafilatura_config():
    custom = use_config()
    custom.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")
    return custom


# -------------------------
# Junk page detector
# -------------------------

def is_junk_content(text: str, title: str = "") -> bool:
    # FIXED: guard against None inputs
    if not text:
        return True
    if not isinstance(text, str):
        return True

    title = title or ""   # FIXED: ensure title is never None
    combined = (text + " " + title).lower()

    if len(text.split()) < 80:
        return True

    for signal in JUNK_SIGNALS:
        if signal in combined:
            return True

    return False


# -------------------------
# Safe metadata extractor
# -------------------------

def safe_get_title(html: str) -> str:
    # FIXED: fully guarded metadata extraction
    try:
        meta = trafilatura.extract_metadata(html)
        if meta is None:
            return ""
        return meta.title or ""
    except Exception:
        return ""


# -------------------------
# Extract text from raw HTML
# -------------------------

def extract_from_html(html: str) -> dict | None:
    if not html or len(html) < 200:
        return None

    try:
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
            config=get_trafilatura_config(),
        )

        # FIXED: guard None text before any use
        if not text or not isinstance(text, str) or len(text.strip()) < 100:
            return None

        title = safe_get_title(html)   # FIXED: use safe extractor

        if is_junk_content(text, title):
            print(f"[web_processor] Junk content detected, skipping.")
            return None

        return {"title": title, "text": text}

    except Exception as e:
        print(f"[web_processor] HTML extraction error: {e}")
        return None


# -------------------------
# Strategy 1: Direct fetch
# -------------------------

def fetch_direct(url: str) -> str | None:
    try:
        downloaded = trafilatura.fetch_url(url, config=get_trafilatura_config())
        if downloaded and len(downloaded) > 200:
            return downloaded
    except Exception as e:
        print(f"[web_processor] Direct fetch failed: {e}")
    return None


# -------------------------
# Strategy 2: Googlebot spoof
# -------------------------

def fetch_as_googlebot(url: str) -> str | None:
    try:
        r = requests.get(url, headers=GOOGLEBOT_HEADERS, timeout=15)
        if r.status_code == 200 and len(r.text) > 200:
            return r.text
    except Exception as e:
        print(f"[web_processor] Googlebot fetch failed: {e}")
    return None


# -------------------------
# Strategy 3: 12ft.io
# -------------------------

def fetch_from_12ft(url: str) -> str | None:
    try:
        r = requests.get(
            f"https://12ft.io/proxy?q={url}",
            headers=BROWSER_HEADERS,
            timeout=20,
            verify=False,
        )
        if r.status_code == 200 and len(r.text) > 200:
            return r.text
    except Exception as e:
        print(f"[web_processor] 12ft fetch failed: {e}")
    return None


# -------------------------
# Strategy 4: Wayback Machine
# -------------------------

def fetch_from_wayback(url: str) -> str | None:
    try:
        check = requests.get(
            f"https://archive.org/wayback/available?url={url}",
            timeout=10
        )
        data = check.json()
        snapshot = data.get("archived_snapshots", {}).get("closest", {})

        if snapshot.get("available"):
            r = requests.get(snapshot["url"], headers=BROWSER_HEADERS, timeout=20)
            if r.status_code == 200 and len(r.text) > 200:
                return r.text
    except Exception as e:
        print(f"[web_processor] Wayback fetch failed: {e}")
    return None


# -------------------------
# Strategy 5: archive.ph
# -------------------------

def fetch_from_archive_ph(url: str) -> str | None:
    try:
        r = requests.get(
            f"https://archive.ph/newest/{url}",
            headers=BROWSER_HEADERS,
            timeout=15,
            allow_redirects=True
        )
        if r.status_code == 200 and len(r.text) > 200:
            return r.text
    except Exception as e:
        print(f"[web_processor] archive.ph fetch failed: {e}")
    return None


# -------------------------
# Strategy 6: Google Cache
# -------------------------

def fetch_from_google_cache(url: str) -> str | None:
    try:
        cache_url = f"https://webcache.googleusercontent.com/search?q=cache:{url}&hl=en"
        r = requests.get(cache_url, headers=BROWSER_HEADERS, timeout=15)
        if r.status_code == 200 and len(r.text) > 200:
            return r.text
    except Exception as e:
        print(f"[web_processor] Google cache fetch failed: {e}")
    return None


# -------------------------
# Full fallback chain
# -------------------------

def fetch_with_fallback(url: str) -> dict | None:
    strategies = [
        ("direct",       fetch_direct),
        ("googlebot",    fetch_as_googlebot),
        ("12ft",         fetch_from_12ft),
        ("wayback",      fetch_from_wayback),
        ("archive.ph",   fetch_from_archive_ph),
        ("google_cache", fetch_from_google_cache),
    ]

    for name, strategy in strategies:
        print(f"[web_processor] Trying strategy: {name}")

        try:
            html = strategy(url)
        except Exception as e:
            # FIXED: catch unexpected crash in any strategy
            print(f"[web_processor] Strategy {name} crashed: {e}")
            continue

        if not html:
            continue

        result = extract_from_html(html)
        if result:
            print(f"[web_processor] Success via: {name}")
            return result

        print(f"[web_processor] Strategy {name} returned junk, skipping.")

    print(f"[web_processor] All strategies failed for: {url}")
    return None


# -------------------------
# Clean text
# -------------------------

def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# -------------------------
# Main processor
# -------------------------

def process_web(url: str) -> dict | None:
    result = fetch_with_fallback(url)

    if not result:
        return None

    content = clean_text(result["text"])

    truncated = False
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS]
        truncated = True

    return {
        "url": url,
        "title": result["title"],
        "content": content,
        "word_count": len(content.split()),
        "truncated": truncated,
    }
