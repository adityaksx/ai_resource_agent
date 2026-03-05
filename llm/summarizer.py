import requests
import textwrap


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"


# -------------------------
# Call LLM
# -------------------------

def call_llm(prompt):

    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 500
            }
        },
        timeout=120
    )

    return r.json()["response"]


# -------------------------
# Chunk long text
# -------------------------

def chunk_text(text, max_chars=3000):

    return textwrap.wrap(text, max_chars)


# -------------------------
# Build prompt
# -------------------------

def build_prompt(text):

    return f"""
You are an AI that extracts useful knowledge from learning resources.

Read the content carefully and extract structured information.

Return output in this format:

TITLE:
A short descriptive title.

SUMMARY:
Explain the resource in 3–5 sentences.

KEY TOPICS:
- topic 1
- topic 2
- topic 3

DIFFICULTY:
Beginner / Intermediate / Advanced

RESOURCE TYPE:
Tutorial / Course / Tool / GitHub Project / Article / Video

CONTENT:
{text}
"""


# -------------------------
# Summarize
# -------------------------

def summarize(text):

    chunks = chunk_text(text)

    summaries = []

    for chunk in chunks:

        prompt = build_prompt(chunk)

        result = call_llm(prompt)

        summaries.append(result)

    # Merge summaries
    combined = "\n\n".join(summaries)

    if len(chunks) > 1:

        final_prompt = f"""
Combine the following summaries into one final structured summary.

{combined}
"""

        return call_llm(final_prompt)

    return combined