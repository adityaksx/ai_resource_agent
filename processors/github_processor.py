import os
import subprocess
import requests

REPO_DIR = "storage/repos"

# Set GITHUB_TOKEN env var to avoid rate limits (60 → 5000 req/hr)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", None)


# -------------------------
# Extract repo info
# -------------------------

def parse_repo_url(url: str) -> tuple[str, str]:
    url = url.strip("/").removesuffix(".git")  # handles repo.git URLs
    parts = url.split("/")
    return parts[-2], parts[-1]


# -------------------------
# Clone or update repository
# -------------------------

def clone_repo(url: str) -> str | None:
    os.makedirs(REPO_DIR, exist_ok=True)

    repo_name = url.strip("/").removesuffix(".git").split("/")[-1]
    path = os.path.join(REPO_DIR, repo_name)

    try:
        if os.path.exists(path):
            # Refresh stale clone
            subprocess.run(
                ["git", "-C", path, "pull"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30
            )
        else:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", url, path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=60
            )
            if result.returncode != 0:
                return None  # clone failed
    except subprocess.TimeoutExpired:
        return None

    return path


# -------------------------
# Read README
# -------------------------

def read_readme(path: str) -> str:
    if not path:
        return ""

    for filename in ["README.md", "readme.md", "Readme.md", "README.txt"]:
        readme_path = os.path.join(path, filename)
        if os.path.exists(readme_path):
            with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    return ""


# -------------------------
# Extract overview from README
# -------------------------

def extract_overview(text: str, max_chars: int = 2000) -> str:
    if not text:
        return ""
    return text[:max_chars]


# -------------------------
# GitHub API metadata
# -------------------------

def get_repo_metadata(url: str) -> dict:
    owner, repo = parse_repo_url(url)
    api = f"https://api.github.com/repos/{owner}/{repo}"

    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    try:
        r = requests.get(api, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        return {
            "name": data.get("name"),
            "description": data.get("description"),
            "stars": data.get("stargazers_count"),
            "language": data.get("language"),
            "topics": data.get("topics", []),
        }
    except Exception as e:
        print(f"[github_processor] metadata fetch failed: {e}")
        return {}


# -------------------------
# Main processor
# -------------------------

def process_github(url: str) -> dict:
    metadata = get_repo_metadata(url)
    repo_path = clone_repo(url)

    readme = read_readme(repo_path)
    overview = extract_overview(readme)

    return {
        "repo": metadata.get("name"),
        "description": metadata.get("description"),
        "language": metadata.get("language"),
        "stars": metadata.get("stars"),
        "topics": metadata.get("topics", []),
        "overview": overview,
    }
