"""
llm/prompt_builder.py
---------------------
Builds ALL LLM prompts from cleaned processor output.

Supports all source types:
  - YouTube / Instagram  → transcript, caption, description, comments
  - GitHub               → overview, readme, content
  - Web / Article        → content, title, description
  - Image (OCR)          → ocr_text
  - Plain text / code    → content, text, body, code
  - PDF / document       → content
"""

from __future__ import annotations

# ── Priority order for fields in the prompt ──────────────────────────────────
_PRIORITY_FIELDS = [
    "title",
    "description",
    "caption",
    "ocr_text",         # image OCR — must come before generic content
    "content",
    "body",
    "text",
    "transcript",
    "overview",
    "readme",
    "code",
    "comments",
    "unique_comments",
    "summary",
]

# ── Human-readable labels ─────────────────────────────────────────────────────
_FIELD_LABELS: dict[str, str] = {
    "title":           "Title",
    "description":     "Description",
    "caption":         "Caption",
    "ocr_text":        "Extracted Text (from image/screenshot)",
    "content":         "Article Content",
    "body":            "Content",
    "text":            "Text",
    "transcript":      "Video Transcript",
    "overview":        "Project Overview",
    "readme":          "README",
    "code":            "Code",
    "comments":        "User Comments",
    "unique_comments": "User Comments",
    "summary":         "Summary",
}

# ── Source-specific task instructions ────────────────────────────────────────
_SOURCE_INSTRUCTIONS: dict[str, str] = {
    "youtube_video":    "This is a YouTube video transcript. Focus on the core concepts explained.",
    "youtube_shorts":   "This is a short video. Extract the single key point.",
    "instagram_reel":   "This is an Instagram reel. Identify the topic and message.",
    "instagram_post":   "This is an Instagram post caption. Extract the core message.",
    "github_repo":      "This is a GitHub project. Explain what it does, why it exists, and how it works.",
    "github_file":      "This is a code file. Explain its purpose, key logic, and any important functions.",
    "web":              "This is a web article. Extract the main ideas and actionable insights.",
    "medium_article":   "This is a Medium article. Summarise the main argument and insights.",
    "substack_article": "This is a newsletter. Extract the main argument and key takeaways.",
    "arxiv_paper":      "This is a research paper. Summarise the problem, method, and findings.",
    "reddit_post":      "This is a Reddit post/discussion. Extract the topic and key opinions.",
    "local_image":      (
        "This is OCR-extracted text from an image or screenshot. "
        "Reconstruct what the image was showing and summarise its content."
    ),
    "plain_text":       "This is a user note or plain text. Summarise the key points.",
    "code_snippet":     "This is a code snippet. Explain what it does and identify the language.",
    "json_data":        "This is structured JSON data. Describe what it represents.",
    "markdown":         "This is a markdown document. Extract the main topics and important info.",
    "pdf_document":     "This is extracted text from a PDF document. Summarise the main content.",
}

_DEFAULT_INSTRUCTION = "Understand the content and extract the most important ideas."

# ── Source-specific output format overrides ───────────────────────────────────
# If a source type needs a different output section, add it here.
# Otherwise the default format below is used.
_SOURCE_OUTPUT_FORMAT: dict[str, str] = {
    "github_repo": """\
MAIN IDEA:
(1–2 sentences explaining what this project does)

KEY FEATURES:
- Feature or capability 1
- Feature or capability 2
- Feature or capability 3

TECH STACK:
- Language / framework / tool used

USE CASE:
Who would use this and why?

SUMMARY:
A short paragraph (3–5 sentences) explaining the project purpose and value.\
""",

    "code_snippet": """\
MAIN IDEA:
(1 sentence: what does this code do?)

LANGUAGE:
(Detected programming language)

KEY LOGIC:
- Step or function 1
- Step or function 2
- Step or function 3

SUMMARY:
A short paragraph explaining the code's purpose and how it works.\
""",

    "arxiv_paper": """\
MAIN IDEA:
(1–2 sentences: what problem does this paper solve?)

METHOD:
- Key approach or algorithm used

FINDINGS:
- Result 1
- Result 2

TAGS:
(Comma-separated: e.g. machine learning, NLP, transformers)

SUMMARY:
A short paragraph (3–5 sentences) covering the problem, method, and findings.\
""",

    "local_image": """\
WHAT THE IMAGE SHOWS:
(1–2 sentences describing what was in the image)

KEY INFORMATION:
- Point 1
- Point 2
- Point 3

SUMMARY:
A short paragraph explaining the image content and what can be learned from it.\
""",
}

_DEFAULT_OUTPUT_FORMAT = """\
MAIN IDEA:
(1–2 sentences explaining the core topic)

KEY INSIGHTS:
- Insight 1
- Insight 2
- Insight 3
- Insight 4

IMPORTANT DETAILS:
- Technical ideas, tools, methods, or steps mentioned

TAGS:
(Comma-separated keywords: e.g. Python, AI, tutorial, web scraping)

DIFFICULTY:
(Beginner / Intermediate / Advanced)

SUMMARY:
A short paragraph (3–5 sentences) explaining what this content is about and what can be learned from it.\
"""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_summary_prompt(data: dict) -> str:
    """
    Build an LLM-ready prompt from any cleaned processor output dict.

    Reads ALL known fields (including ocr_text) and any unknown extra
    fields so no content is ever silently dropped.
    """
    if not data:
        return "No content was provided. Please try a different resource."

    source_type  = data.get("source_type", "")
    instruction  = _SOURCE_INSTRUCTIONS.get(source_type, _DEFAULT_INSTRUCTION)
    output_format = _SOURCE_OUTPUT_FORMAT.get(source_type, _DEFAULT_OUTPUT_FORMAT)

    content_parts: list[str] = []
    seen: set[str] = set()

    # Process fields in priority order, then any extras
    all_keys = _PRIORITY_FIELDS + [
        k for k in data
        if k not in _PRIORITY_FIELDS and k != "source_type"
    ]

    for key in all_keys:
        if key in seen or key not in data:
            continue
        seen.add(key)

        value = data[key]
        label = _FIELD_LABELS.get(key, key.replace("_", " ").title())

        if not value:
            continue

        if isinstance(value, list):
            items = "\n".join(f"- {c}" for c in value[:20] if c)
            if items:
                content_parts.append(f"{label}:\n{items}")

        elif isinstance(value, str) and value.strip():
            content_parts.append(f"{label}:\n{value.strip()}")

    if not content_parts:
        return (
            "The content could not be extracted or was empty. "
            "Please try a different resource."
        )

    content = "\n\n".join(content_parts)

    prompt = f"""You are an intelligent knowledge extraction assistant.

Task: {instruction}

Instructions:
- Focus on concepts, facts, insights, and explanations.
- Ignore filler words, ads, sponsors, and navigation text.
- If the content is OCR from a screenshot, reconstruct what it was showing.
- If the content is code, explain what it does clearly.
- Avoid repeating sentences.
- Be specific — mention actual tools, technologies, names, and steps.
- Keep answers concise and useful.

Content:
{content}

Return output in this exact format:

{output_format}
"""
    return prompt


def build_merge_prompt(partial_summaries: list[str]) -> str:
    """
    Build a prompt to merge multiple chunk summaries into one final summary.
    Called by summarizer.py when content was split into chunks.
    """
    combined = "\n\n---\n\n".join(
        f"[Part {i + 1}]\n{s.strip()}" for i, s in enumerate(partial_summaries)
    )

    return f"""You are an intelligent knowledge extraction assistant.

The following are partial summaries of different sections of the same resource.
Combine them into one single, clean, non-repetitive structured summary.

Partial Summaries:
{combined}

Return the final output in this exact format:

MAIN IDEA:
(1–2 sentences explaining the core topic)

KEY INSIGHTS:
- Insight 1
- Insight 2
- Insight 3
- Insight 4

IMPORTANT DETAILS:
- Technical ideas, tools, methods, or steps mentioned

TAGS:
(Comma-separated keywords)

DIFFICULTY:
(Beginner / Intermediate / Advanced)

SUMMARY:
A short paragraph (3–5 sentences) explaining what this content is about and what can be learned from it.
"""
