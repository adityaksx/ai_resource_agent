from urllib.parse import urlparse

def detect_source(url):

    domain = urlparse(url).netloc

    if "youtube.com" in domain or "youtu.be" in domain:
        return "youtube"

    if "instagram.com" in domain:
        return "instagram"

    if "github.com" in domain:
        return "github"

    return "web"