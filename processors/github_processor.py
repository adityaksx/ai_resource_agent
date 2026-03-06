import os
import subprocess
import requests
from urllib.parse import urlparse

REPO_DIR     = "storage/repos"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", None)

_HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}


# ─────────────────────────────────────────────────────────────────────────────
# URL PARSING  (fixed)
# ─────────────────────────────────────────────────────────────────────────────

def parse_github_url(url: str) -> dict:
    """
    Parse any GitHub URL and return its components.

    Returns dict with keys:
      type   : "repo" | "user" | "file" | "gist" | "unknown"
      owner  : str
      repo   : str | None
      path   : str | None  (for file URLs)

    Examples:
      https://github.com/tiangolo/fastapi          → repo
      https://github.com/palinkiewicz              → user
      github.com/tiangolo/fastapi                  → repo  (no https)
      https://github.com/tiangolo/fastapi/blob/... → file
      https://gist.github.com/user/abc123          → gist
    """
    # Normalise — add scheme if missing so urlparse works
    if not url.startswith("http"):
        url = "https://" + url

    parsed = urlparse(url.strip().rstrip("/").removesuffix(".git"))
    host   = parsed.netloc.lower()
    parts  = [p for p in parsed.path.strip("/").split("/") if p]

    # Gist URLs
    if "gist.github.com" in host:
        owner = parts[0] if len(parts) > 0 else ""
        repo  = parts[1] if len(parts) > 1 else ""
        return {"type": "gist", "owner": owner, "repo": repo, "path": None}

    # Must be github.com from here
    if "github.com" not in host:
        return {"type": "unknown", "owner": "", "repo": None, "path": None}

    if len(parts) == 0:
        return {"type": "unknown", "owner": "", "repo": None, "path": None}

    if len(parts) == 1:
        # github.com/username  → user profile
        return {"type": "user", "owner": parts[0], "repo": None, "path": None}

    owner = parts[0]
    repo  = parts[1]

    if len(parts) > 2 and parts[2] in ("blob", "tree", "raw"):
        # github.com/owner/repo/blob/branch/path/to/file
        file_path = "/".join(parts[4:]) if len(parts) > 4 else ""
        return {"type": "file", "owner": owner, "repo": repo, "path": file_path}

    return {"type": "repo", "owner": owner, "repo": repo, "path": None}


# ─────────────────────────────────────────────────────────────────────────────
# GITHUB API
# ─────────────────────────────────────────────────────────────────────────────

def get_repo_metadata(owner: str, repo: str) -> dict:
    """Fetch repo metadata from GitHub API."""
    api = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        r = requests.get(api, headers=_HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        return {
            "name":        data.get("name",              ""),
            "description": data.get("description",       ""),
            "stars":       data.get("stargazers_count",  0),
            "forks":       data.get("forks_count",       0),
            "language":    data.get("language",          ""),
            "topics":      data.get("topics",            []),
            "license":     (data.get("license") or {}).get("name", ""),
            "updated_at":  data.get("updated_at",        ""),
            "open_issues": data.get("open_issues_count", 0),
        }
    except requests.exceptions.HTTPError as e:
        if "404" in str(e):
            print(f"[github_processor] Repo not found: {owner}/{repo}")
        else:
            print(f"[github_processor] API error: {e}")
        return {}
    except Exception as e:
        print(f"[github_processor] Metadata fetch failed: {e}")
        return {}


def get_languages(owner: str, repo: str) -> list[str]:
    """Fetch all languages used in the repo."""
    api = f"https://api.github.com/repos/{owner}/{repo}/languages"
    try:
        r = requests.get(api, headers=_HEADERS, timeout=10)
        r.raise_for_status()
        return list(r.json().keys())      # ["Python", "JavaScript", ...]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# CLONE + README
# ─────────────────────────────────────────────────────────────────────────────

def clone_repo(url: str, repo_name: str) -> str | None:
    """Clone repo shallowly. Returns local path or None on failure."""
    os.makedirs(REPO_DIR, exist_ok=True)
    path = os.path.join(REPO_DIR, repo_name)

    # Ensure URL has https:// prefix for git
    if not url.startswith("http"):
        url = "https://" + url

    try:
        if os.path.exists(path):
            subprocess.run(
                ["git", "-C", path, "pull"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
        else:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", url, path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=60,
            )
            if result.returncode != 0:
                print(f"[github_processor] Clone failed for: {url}")
                return None
    except subprocess.TimeoutExpired:
        print(f"[github_processor] Clone timed out: {url}")
        return None

    return path


def read_readme(path: str) -> str:
    if not path:
        return ""
    for filename in ["README.md", "readme.md", "Readme.md", "README.txt", "README.rst"]:
        readme_path = os.path.join(path, filename)
        if os.path.exists(readme_path):
            with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    return ""

def read_source_files(path: str, max_chars: int = 2000) -> str:
    """
    Fallback when no README exists — reads actual source files
    to give the LLM real content instead of hallucinating.
    Reads: package.json → index.html → main JS file → style.css
    """
    if not path:
        return ""

    collected = []
    total     = 0

    # Priority order — most descriptive files first
    candidates = [
        "package.json",
        "pyproject.toml",
        "index.html",
        "main.py",
        "app.py",
        "script.js",
        "main.js",
        "index.js",
        "src/index.js",
        "src/main.js",
        "src/App.jsx",
        "src/App.tsx",
        "style.css",
    ]

    for filename in candidates:
        fpath = os.path.join(path, filename)
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()[:600]   # 600 chars per file max
            collected.append(f"=== {filename} ===\n{content}")
            total += len(content)
            if total >= max_chars:
                break
        except Exception:
            continue

    return "\n\n".join(collected)


def extract_overview(text: str, max_chars: int = 2500) -> str:
    if not text:
        return ""
    return text[:max_chars]


# ─────────────────────────────────────────────────────────────────────────────
# GITHUB USER PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def get_user_metadata(username: str) -> dict:
    api = f"https://api.github.com/users/{username}"
    try:
        r = requests.get(api, headers=_HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        return {
            "login":        data.get("login",        ""),
            "name":         data.get("name",         ""),
            "bio":          data.get("bio",          ""),
            "company":      data.get("company",      ""),
            "location":     data.get("location",     ""),
            "blog":         data.get("blog",         ""),
            "followers":    data.get("followers",    0),
            "following":    data.get("following",    0),
            "public_repos": data.get("public_repos", 0),
        }
    except Exception as e:
        print(f"[github_processor] User metadata failed: {e}")
        return {}


def get_user_top_repos(username: str, limit: int = 6) -> list[dict]:
    api = f"https://api.github.com/users/{username}/repos?sort=stars&per_page={limit}"
    try:
        r = requests.get(api, headers=_HEADERS, timeout=10)
        r.raise_for_status()
        return [
            {
                "name":        repo.get("name",             ""),
                "description": repo.get("description",      ""),
                "language":    repo.get("language",         ""),
                "stars":       repo.get("stargazers_count", 0),
                "topics":      repo.get("topics",           []),
            }
            for repo in r.json()
            if not repo.get("fork")
        ]
    except Exception as e:
        print(f"[github_processor] User repos failed: {e}")
        return []


def process_github_user(url: str, username: str) -> dict | None:
    user  = get_user_metadata(username)
    repos = get_user_top_repos(username)

    if not user:
        print(f"[github_processor] No user data for: {username}")
        return None

    return {
        "source_type":  "github_user",
        "url":          url,
        "username":     user.get("login",        ""),
        "name":         user.get("name",         ""),
        "bio":          user.get("bio",          ""),
        "company":      user.get("company",      ""),
        "location":     user.get("location",     ""),
        "blog":         user.get("blog",         ""),
        "followers":    user.get("followers",    0),
        "following":    user.get("following",    0),
        "public_repos": user.get("public_repos", 0),
        "top_repos":    repos,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

def process_github(url: str) -> dict | None:
    """
    Routes GitHub URLs by type:
      repo  → API metadata + clone + README
      file  → API metadata only (no clone needed)
      gist  → API metadata only
      user  → returns None (handled as unsupported in main.py)
    """
    parsed = parse_github_url(url)
    kind   = parsed["type"]
    owner  = parsed["owner"]
    repo   = parsed["repo"]

    print(f"[github_processor] Detected type={kind}  owner={owner}  repo={repo}")

    # ── User profile — not a repo, cannot process ─────────────────────────
    if kind == "user":
        print(f"[github_processor] User profile URL → fetching via API")
        return process_github_user(url, owner)


    # ── Unknown ───────────────────────────────────────────────────────────
    if kind == "unknown" or not owner or not repo:
        print(f"[github_processor] Unrecognised GitHub URL: {url}")
        return None

    # ── Repo / File / Gist — fetch metadata ───────────────────────────────
    metadata  = get_repo_metadata(owner, repo)
    languages = get_languages(owner, repo)

    if not metadata:
        print(f"[github_processor] No metadata returned for {owner}/{repo}")
        return None

    # ── Clone and read README (repo only) ─────────────────────────────────
    overview = ""
    if kind == "repo":
        repo_path = clone_repo(url, repo)
        readme    = read_readme(repo_path)
        if readme:
            overview = extract_overview(readme)
        else:
            print(f"[github_processor] No README found — reading source files")
            source_content = read_source_files(repo_path)
            if source_content:
                overview = "[No README. Summarize strictly from source files below. Do NOT invent features.]\n\n" + source_content
            else:
                print(f"[github_processor] No source content found for {owner}/{repo}")

    return {
        "source_type": f"github_{kind}",
        "url":         url,
        "repo":        metadata.get("name",        ""),
        "owner":       owner,
        "description": metadata.get("description", ""),
        "language":    metadata.get("language",     ""),
        "languages":   languages,
        "stars":       metadata.get("stars",        0),
        "forks":       metadata.get("forks",        0),
        "topics":      metadata.get("topics",       []),
        "license":     metadata.get("license",      ""),
        "open_issues": metadata.get("open_issues",  0),
        "updated_at":  metadata.get("updated_at",   ""),
        "overview":    overview,
        "has_readme":  bool(readme) if kind == "repo" else False,   # ← ADD THIS
    }
