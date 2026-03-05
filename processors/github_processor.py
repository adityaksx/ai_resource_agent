import subprocess
import os


def clone_repo(url):

    repo_name = url.split("/")[-1]

    path = f"storage/repos/{repo_name}"

    subprocess.run(["git", "clone", url, path])

    return path


def read_readme(path):

    readme = os.path.join(path, "README.md")

    if os.path.exists(readme):
        with open(readme, "r", encoding="utf-8") as f:
            return f.read()

    return ""