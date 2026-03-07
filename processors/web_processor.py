import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
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

# ── Domains known to require JavaScript rendering ─────────────────────────────
# These skip strategies 1-6 entirely and go straight to Playwright
_JS_HEAVY_DOMAINS = {
    "vercel.app", "netlify.app", "phet.colorado.edu",
    "react.dev", "nextjs.org", "svelte.dev",
    "angular.io", "vuejs.org", "remix.run",
    "codesandbox.io", "stackblitz.com",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_trafilatura_config():
    custom = use_config()
    custom.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")
    return custom


def _is_js_heavy(url: str) -> bool:
    """Check if URL is from a known JS-heavy domain — skip to Playwright immediately."""
    from urllib.parse import urlparse
    try:
        host = urlparse(url).netloc.lower().lstrip("www.")
        return any(host == d or host.endswith("." + d) for d in _JS_HEAVY_DOMAINS)
    except Exception:
        return False


def is_junk_content(text: str, title: str = "", min_words: int = 60) -> bool:
    if not text or not isinstance(text, str):
        return True
    title      = title or ""
    combined   = (text + " " + title).lower()
    word_count = len(text.split())

    # Raised threshold: JS sites often render sparse content
    if word_count < min_words:
        return True

    for signal in JUNK_SIGNALS:
        if signal in combined:
            return True

    return False


def safe_get_title(html: str) -> str:
    try:
        meta = trafilatura.extract_metadata(html)
        if meta is None:
            return ""
        return meta.title or ""
    except Exception:
        return ""


def extract_from_html(html: str, min_words: int = 60) -> dict | None:
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
        if not text or not isinstance(text, str) or len(text.strip()) < min_words // 2:
            return None

        title = safe_get_title(html)

        if is_junk_content(text, title, min_words=min_words):
            print(f"[web_processor] Junk content detected, skipping.")
            return None

        return {"title": title, "text": text}
    except Exception as e:
        print(f"[web_processor] HTML extraction error: {e}")
        return None


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGIES 1–6  (unchanged — static HTML fetchers)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_direct(url: str) -> str | None:
    try:
        downloaded = trafilatura.fetch_url(url, config=get_trafilatura_config())
        if downloaded and len(downloaded) > 200:
            return downloaded
    except Exception as e:
        print(f"[web_processor] Direct fetch failed: {e}")
    return None


def fetch_as_googlebot(url: str) -> str | None:
    try:
        r = requests.get(url, headers=GOOGLEBOT_HEADERS, timeout=15)
        if r.status_code == 200 and len(r.text) > 200:
            return r.text
    except Exception as e:
        print(f"[web_processor] Googlebot fetch failed: {e}")
    return None


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


def fetch_from_wayback(url: str) -> str | None:
    try:
        check    = requests.get(
            f"https://archive.org/wayback/available?url={url}", timeout=10
        )
        data     = check.json()
        snapshot = data.get("archived_snapshots", {}).get("closest", {})
        if snapshot.get("available"):
            r = requests.get(snapshot["url"], headers=BROWSER_HEADERS, timeout=20)
            if r.status_code == 200 and len(r.text) > 200:
                return r.text
    except Exception as e:
        print(f"[web_processor] Wayback fetch failed: {e}")
    return None


def fetch_from_archive_ph(url: str) -> str | None:
    try:
        r = requests.get(
            f"https://archive.ph/newest/{url}",
            headers=BROWSER_HEADERS,
            timeout=15,
            allow_redirects=True,
        )
        if r.status_code == 200 and len(r.text) > 200:
            return r.text
    except Exception as e:
        print(f"[web_processor] archive.ph fetch failed: {e}")
    return None


def fetch_from_google_cache(url: str) -> str | None:
    try:
        cache_url = f"https://webcache.googleusercontent.com/search?q=cache:{url}&hl=en"
        r = requests.get(cache_url, headers=BROWSER_HEADERS, timeout=15)
        if r.status_code == 200 and len(r.text) > 200:
            return r.text
    except Exception as e:
        print(f"[web_processor] Google cache fetch failed: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 7 — Playwright (headless Chromium, renders JavaScript)
# ─────────────────────────────────────────────────────────────────────────────

# Dedicated thread — has its own event loop, bypasses Windows SelectorEventLoop
# subprocess restriction that causes NotImplementedError
_playwright_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="playwright")


def _playwright_sync(url: str) -> str | None:
    """
    Runs sync Playwright inside a dedicated thread.
    Must NOT be async — threads have their own event loop.
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ]
            )
            context = browser.new_context(
                user_agent=BROWSER_HEADERS["User-Agent"],
                locale="en-US",
                viewport={"width": 1280, "height": 800},
            )
            page = context.new_page()

            page.route(
                "**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf,mp4,mp3}",
                lambda route: route.abort()
            )

            try:
                page.goto(url, wait_until="networkidle", timeout=30_000)
            except PWTimeout:
                print(f"[web_processor] Playwright: networkidle timeout, using current state")

            page.wait_for_timeout(2000)
            html = page.content()
            browser.close()

            if html and len(html) > 200:
                print(f"[web_processor] Playwright: got {len(html)} chars of HTML")
                return html

    except ImportError:
        print(
            "[web_processor] Playwright not installed.\n"
            "Run: pip install playwright && playwright install chromium"
        )
    except Exception as e:
        print(f"[web_processor] Playwright fetch failed: {e}")

    return None


async def fetch_with_playwright(url: str) -> str | None:
    """
    Async wrapper — offloads _playwright_sync to a dedicated thread.
    Fixes Windows NotImplementedError: SelectorEventLoop cannot spawn subprocesses.
    All callers use await fetch_with_playwright(url) — interface unchanged.
    """
    print(f"[web_processor] Playwright: launching headless browser...")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_playwright_executor, _playwright_sync, url)

# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK CHAIN
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_with_fallback(url: str) -> dict | None:
    if _is_js_heavy(url):
        print(f"[web_processor] JS-heavy domain detected, using Playwright directly")
        html   = await fetch_with_playwright(url)          # ← await
        result = extract_from_html(html, min_words=25) if html else None   # ← min_words=25
        if result:
            print(f"[web_processor] Success via: playwright (direct)")
            return result
        print(f"[web_processor] Playwright failed for JS-heavy site: {url}")
        return None

    strategies = [
        ("direct",       fetch_direct),
        ("googlebot",    fetch_as_googlebot),
        ("12ft",         fetch_from_12ft),
        ("wayback",      fetch_from_wayback),
        ("archive.ph",   fetch_from_archive_ph),
        ("google_cache", fetch_from_google_cache),
        ("playwright",   fetch_with_playwright),
    ]

    for name, strategy in strategies:
        print(f"[web_processor] Trying strategy: {name}")
        try:
            # Playwright is async, others are sync — handle both
            if name == "playwright":
                html = await strategy(url)                 # ← await only for playwright
            else:
                html = strategy(url)                       # sync call for others
        except Exception as e:
            print(f"[web_processor] Strategy {name} crashed: {e}")
            continue

        if not html:
            continue

        result = extract_from_html(html, min_words=25 if name == "playwright" else 60)
        if result:
            print(f"[web_processor] Success via: {name}")
            return result

        print(f"[web_processor] Strategy {name} returned junk, skipping.")

    print(f"[web_processor] All strategies failed for: {url}")
    return None

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

async def process_web(url: str) -> dict | None:
    result = await fetch_with_fallback(url)                # ← await

    if not result:
        return None

    content = clean_text(result["text"])

    truncated = False
    if len(content) > MAX_CONTENT_CHARS:
        content   = content[:MAX_CONTENT_CHARS]
        truncated = True

    return {
        "url":        url,
        "title":      result["title"],
        "content":    content,
    }