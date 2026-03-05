import requests


def summarize(text):

    prompt = f"""
Summarize this developer resource.

Return:
Title
Summary
Tags
Difficulty
Learning value

Content:
{text}
"""

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )

    return r.json()["response"]