import os
import subprocess
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import git


RAW_DIR = "storage/raw"
VIDEO_DIR = "storage/videos"
IMAGE_DIR = "storage/images"
REPO_DIR = "storage/repos"


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(REPO_DIR, exist_ok=True)


# -------------------------
# YOUTUBE SHORTS
# -------------------------

def download_youtube_shorts(url):
    """
    Download only YouTube Shorts using yt-dlp
    """

    ensure_dirs()

    cmd = [
        "yt-dlp",
        "-S", "res:480",
        "-o", f"{VIDEO_DIR}/%(title)s.%(ext)s",
        url
    ]

    subprocess.run(cmd)

    return "YouTube Shorts downloaded"


# -------------------------
# INSTAGRAM
# -------------------------

def download_instagram(url):
    """
    Download Instagram post / reel
    """

    ensure_dirs()

    cmd = [
        "yt-dlp",
        "-o",
        f"{VIDEO_DIR}/%(title)s.%(ext)s",
        url
    ]

    subprocess.run(cmd)

    return "Instagram media downloaded"


# -------------------------
# GITHUB REPOSITORY
# -------------------------

def download_github_repo(url):
    """
    Clone GitHub repository
    """

    ensure_dirs()

    repo_name = url.rstrip("/").split("/")[-1]
    path = os.path.join(REPO_DIR, repo_name)

    if os.path.exists(path):
        return f"Repo already exists: {path}"

    git.Repo.clone_from(url, path)

    return f"Repo cloned to {path}"


# -------------------------
# WEB ARTICLE
# -------------------------

def download_webpage(url):
    """
    Download webpage and extract readable text
    """

    ensure_dirs()

    response = requests.get(url, timeout=10)

    soup = BeautifulSoup(response.text, "html.parser")

    paragraphs = soup.find_all("p")

    text = "\n".join([p.get_text() for p in paragraphs])

    file_name = urlparse(url).netloc.replace(".", "_") + ".txt"

    path = os.path.join(RAW_DIR, file_name)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    return path


# -------------------------
# ROUTER
# -------------------------

def download(url, source):
    """
    Main downloader router
    """

    if source.startswith("youtube"):
        return download_youtube_shorts(url)

    if source.startswith("instagram"):
        return download_instagram(url)

    if source.startswith("github"):
        return download_github_repo(url)

    if source == "web":
        return download_webpage(url)

    return "Unsupported source"