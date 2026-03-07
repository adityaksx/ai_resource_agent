"""
llm/prompt_builder.py
---------------------
Builds ALL LLM prompts used anywhere in the pipeline.

Prompt functions (public API):
  build_summary_prompt(data)           → Stage 4 final answer
  build_merge_prompt(summaries)        → Stage 4 multi-chunk merge
  build_classifier_prompt(input_str)   → Stage 1 classify input type
  build_guidance_prompt(input, type)   → Stage 2 extraction guidance
  build_enrich_prompt(data, guidance)  → Stage 3 fill data gaps

Design rules:
  - ALL prompt strings live here — zero prompt logic in other files.
  - No LLM calls here — only string construction.
  - No imports from other project files — pure Python stdlib only.

Called by:
  llm/summarizer.py    → build_summary_prompt, build_merge_prompt
  llm/pipeline.py      → build_classifier_prompt, build_guidance_prompt,
                          build_enrich_prompt
"""

from __future__ import annotations

import json


# ─────────────────────────────────────────────────────────────────────────────
# FIELD CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Priority order for content fields in the prompt body.
# Fields listed first appear first in the prompt — order matters.
_PRIORITY_FIELDS: list[str] = [
    "title",
    "description",
    "caption",
    "ocr_text",
    "content",
    "body",
    "text",
    "transcript",
    "overview",
    "readme",
    "code",
    "top_comments",       # ← ADD (YouTube + Instagram)
    "recent_captions",    # ← ADD (Instagram profile)
    "comments",
    "unique_comments",
    "summary",
    # enriched
    "inferred_audience",
    "inferred_difficulty",
    "related_tools",
    "missing_context",
]


# Human-readable labels shown in the prompt
_FIELD_LABELS: dict[str, str] = {
    "title":               "Title",
    "description":         "Description",
    "caption":             "Caption",
    "ocr_text":            "Extracted Text (from image/screenshot)",
    "content":             "Article Content",
    "body":                "Content",
    "text":                "Text",
    "transcript":          "Video Transcript",
    "overview":            "Project Overview",
    "code":                "Code",
    "comments":            "User Comments",
    "unique_comments":     "User Comments",
    "summary":             "Summary",
    # Pipeline enriched fields
    "inferred_audience":   "Target Audience (inferred)",
    "inferred_difficulty": "Difficulty Level (inferred)",
    "related_tools":       "Related Tools / Alternatives",
    "missing_context":     "Additional Context",
    "top_comments":     "Top Comments",          # ← ADD
    "recent_captions":  "Recent Posts",          # ← ADD
    "language":         "Programming Language",  # ← ADD
    "languages":        "Languages Used",        # ← ADD
    "tags":             "Tags",                  # ← ADD
    "topics":           "Topics",                # ← ADD
    "uploader":         "Posted By",             # ← ADD
    "username":         "Username",              # ← ADD
    "full_name":        "Name",                  # ← ADD
    "bio":              "Bio",                   # ← ADD
    "category":         "Category",             # ← ADD
    "top_repos":        "Top Repositories",      # ← ADD
}

# Pure metadata — never shown in the content block, only used for routing.
# These are preserved by cleaner.py but excluded here from the prompt body.
_META_FIELDS: set[str] = {
    "source_type", "url", "video_id", "channel",
    "author", "date", "source",
    # numeric stats — never content
    "char_count", "word_count", "line_count", "truncated",
    "view_count", "duration", "upload_date", "likes",
    "stars", "forks", "open_issues", "updated_at",
    "followers", "following", "posts", "public_repos",
    "is_verified", "is_business",
    # file paths — never content
    "image_path", "filename", "repo_path",
    # internal routing
    "has_readme", "files",
}


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE-SPECIFIC TASK INSTRUCTIONS
# ─────────────────────────────────────────────────────────────────────────────

_SOURCE_INSTRUCTIONS: dict[str, str] = {
    # YouTube
    "youtube_video":    "This is a YouTube video transcript. Focus on the core concepts, steps, and takeaways explained.",
    "youtube_shorts":   "This is a short-form video (under 60s). Extract the single key point or message.",
    "youtube_playlist": "This is a YouTube playlist. Describe the overall theme and what topics the series covers.",
    "youtube_channel":  "This is a YouTube channel. Describe the creator's niche, content style, and target audience.",
    "youtube_live":     "This is a YouTube live stream transcript. Extract the key topics discussed.",
    # Instagram
    "instagram_reel":   "This is an Instagram reel caption/transcript. Identify the core topic and message.",
    "instagram_post":   "This is an Instagram post caption. Extract the core message and any calls to action.",
    # GitHub
    "github_repo":      "This is a GitHub repository. Explain what it does, why it exists, its tech stack, and how to use it.",
    "github_file":      "This is a single code file from GitHub. Explain its purpose, key functions, and where it fits in a larger project.",
    "github_gist":      "This is a GitHub Gist (code snippet or note). Explain what it does and when you would use it.",
    # Web / Publishing
    "web":              "This is a web article or page. Extract the main ideas, actionable insights, and any important details.",
    "medium_article":   "This is a Medium article. Summarise the main argument, supporting points, and practical takeaways.",
    "substack_article": "This is a Substack newsletter. Extract the main argument, key claims, and author's perspective.",
    "notion_page":      "This is a Notion page or document. Extract the main topic, structure, and key information.",
    "reddit_post":      "This is a Reddit post or thread. Extract the topic, main question/claim, and key community opinions.",
    "reddit_subreddit": "This is a Reddit community page. Describe the subreddit's topic and what kind of content it covers.",
    # Research / Documents
    "arxiv_paper":      "This is an academic research paper. Summarise the problem, proposed method, experiments, and findings.",
    "pdf_document":     "This is extracted text from a PDF document. Summarise the main content, structure, and key points.",
    # AI / ML
    "huggingface_model":   "This is a HuggingFace model card. Explain what the model does, what task it solves, and how to use it.",
    "huggingface_dataset": "This is a HuggingFace dataset card. Explain what data it contains and what it is used for.",
    "huggingface_space":   "This is a HuggingFace Space (demo app). Explain what it demonstrates and how to interact with it.",
    # Video (non-YouTube)
    "loom_video":       "This is a Loom screen recording. Extract the main topic being demonstrated or explained.",
    "vimeo_video":      "This is a Vimeo video. Extract the main topic, theme, and key information.",
    # Local / raw
    "local_image":      "This is OCR-extracted text from an image or screenshot. Reconstruct what the image was showing and summarise its content.",
    "plain_text":       "This is a user note or plain text. Summarise the key points and any important information.",
    "news_headline":    "This is a news headline or short news snippet. Identify the event, key parties involved, and significance.",
    "code_snippet":     "This is a code snippet. Explain what it does, identify the language, and describe the key logic.",
    "notebook":         "This is a Jupyter notebook. Explain the analysis goal, methods used, and key findings or outputs.",
    "json_data":        "This is structured JSON data. Describe what it represents and what each key section contains.",
    "markdown":         "This is a markdown document. Extract the main topics, structure, and important information.",
    "youtube_channel":       "This is a YouTube channel page. Describe the creator's content niche, upload frequency, and target audience based on available metadata.",
    "loom_video":            "This is a Loom screen recording. Extract the main topic being demonstrated, key steps shown, and who this recording is intended for.",
    "vimeo_video":           "This is a Vimeo video. Extract the main topic, creative intent, and key information presented.",
    "substack_article":      "This is a Substack newsletter post. Extract the author's main argument, key claims, supporting evidence, and the author's perspective.",
    "notion_page":           "This is a Notion page or document. Extract the main topic, structure, and all key information present.",
    "huggingface_dataset":   "This is a HuggingFace dataset card. Explain what data it contains, its format, size, intended use cases, and any known limitations.",
    "huggingface_space":     "This is a HuggingFace Space demo app. Explain what model or task it demonstrates, how to interact with it, and what it produces.",
    "pdf_document":          "This is extracted text from a PDF document. Identify the document type, summarise the main content, structure, and key points.",
    "plain_text":            "This is a user note or pasted text. Identify the topic and summarise all key points, facts, and actionable information.",
    "json_data":             "This is structured JSON data. Describe what it represents, what the top-level keys mean, and what the data could be used for.",
    "markdown":              "This is a markdown document. Extract the main topics by heading, key bullet points, and any important details.",
    "instagram_profile":     "This is an Instagram profile. Describe the creator's niche, content style, audience size, and what their recent posts are about.",
    "github_user":           "This is a GitHub user profile. Describe the developer's background, main projects, primary languages, and areas of expertise.",
    "news_headline":         "This is a news headline or short snippet. Identify the event, key parties involved, timeline, and why it matters.",

}

_DEFAULT_INSTRUCTION = "Understand the content and extract the most important ideas, facts, and insights."

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPER — render a list[dict] as readable bullet lines
# ─────────────────────────────────────────────────────────────────────────────

def _render_list_of_dicts(items: list[dict], max_items: int = 10) -> str:
    """
    Renders a list of dicts (e.g. top_repos, recent_captions as dicts)
    into readable bullet lines for the prompt.

    Tries common keys: name, title, description, caption, language, stars.
    Falls back to key=value pairs for unknown structures.
    """
    lines = []
    for item in items[:max_items]:
        if not isinstance(item, dict):
            lines.append(f"- {item}")
            continue

        name  = item.get("name")  or item.get("title")   or ""
        desc  = item.get("description") or item.get("caption") or ""
        lang  = item.get("language") or ""
        stars = item.get("stars")

        parts = []
        if name:  parts.append(name)
        if desc:  parts.append(desc[:120])
        if lang:  parts.append(f"[{lang}]")
        if stars: parts.append(f"⭐ {stars}")

        if parts:
            lines.append("- " + " — ".join(parts))
        else:
            # Unknown dict shape — render as key: value pairs
            kv = ", ".join(f"{k}: {str(v)[:60]}" for k, v in item.items() if v)
            if kv:
                lines.append(f"- {kv}")

    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE-SPECIFIC OUTPUT FORMATS
# ─────────────────────────────────────────────────────────────────────────────

_SOURCE_OUTPUT_FORMAT: dict[str, str] = {

    "github_repo": """\
MAIN IDEA:
(1–2 sentences explaining what this project does)

KEY FEATURES:
- Feature or capability 1
- Feature or capability 2
- Feature or capability 3

TECH STACK:
- Language / framework / tool

USE CASE:
Who would use this and why?

TAGS:
(Comma-separated: e.g. Python, API, open-source, CLI)

SUMMARY:
A short paragraph (3–5 sentences) explaining the project purpose, who built it, and its value.\
""",

    "github_file": """\
MAIN IDEA:
(1 sentence: what does this file do?)

LANGUAGE:
(Detected programming language)

KEY FUNCTIONS / CLASSES:
- Function or class 1 — what it does
- Function or class 2 — what it does

DEPENDENCIES:
- External libraries or modules used

SUMMARY:
A short paragraph explaining the file's role and how it fits in a larger project.\
""",

    "github_gist": """\
MAIN IDEA:
(1 sentence: what does this gist do?)

LANGUAGE:
(Detected programming language)

KEY LOGIC:
- Step or pattern 1
- Step or pattern 2

WHEN TO USE:
(One sentence describing the use case)

SUMMARY:
A short paragraph explaining the gist's purpose and how to apply it.\
""",

    "notebook": """\
MAIN IDEA:
(1–2 sentences: what is this notebook analysing or demonstrating?)

METHODS USED:
- Library or technique 1
- Library or technique 2

KEY FINDINGS:
- Finding or output 1
- Finding or output 2

TAGS:
(Comma-separated: e.g. data science, pandas, machine learning)

SUMMARY:
A short paragraph (3–5 sentences) covering the goal, approach, and results of this notebook.\
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

DIFFICULTY:
(Beginner / Intermediate / Advanced)

SUMMARY:
A short paragraph (3–5 sentences) covering the problem, method, and key findings.\
""",

    "local_image": """\
WHAT THE IMAGE SHOWS:
(1–2 sentences describing what was in the image or screenshot)

KEY INFORMATION:
- Point 1
- Point 2
- Point 3

SUMMARY:
A short paragraph explaining the image content and what can be learned from it.\
""",

    "youtube_video": """\
MAIN IDEA:
(1–2 sentences: what is this video about?)

KEY CONCEPTS:
- Concept or step 1
- Concept or step 2
- Concept or step 3

TOOLS / TECHNOLOGIES MENTIONED:
- Tool or tech 1
- Tool or tech 2

TAGS:
(Comma-separated: e.g. tutorial, Python, machine learning)

DIFFICULTY:
(Beginner / Intermediate / Advanced)

SUMMARY:
A short paragraph (3–5 sentences) explaining what the video covers and what viewers will learn.\
""",

    "reddit_post": """\
MAIN IDEA:
(1–2 sentences: what is this post about or asking?)

KEY OPINIONS / ANSWERS:
- View or answer 1
- View or answer 2
- View or answer 3

CONSENSUS:
(1 sentence: what did the community generally agree on, if anything?)

TAGS:
(Comma-separated keywords)

SUMMARY:
A short paragraph summarising the post topic and the most useful community responses.\
""",

    "news_headline": """\
MAIN IDEA:
(1–2 sentences: what happened?)

KEY PARTIES:
- Who is involved?

SIGNIFICANCE:
(1–2 sentences: why does this matter?)

TAGS:
(Comma-separated: e.g. AI, India, startup, policy)

SUMMARY:
A short paragraph explaining the news event and its broader context.\
""",

    "huggingface_model": """\
MAIN IDEA:
(1–2 sentences: what does this model do?)

TASK:
(The ML task: e.g. text classification, image generation, translation)

USAGE:
- How to load or use it (key steps)

TAGS:
(Comma-separated: e.g. NLP, transformers, fine-tuned, PyTorch)

SUMMARY:
A short paragraph explaining the model's purpose, training, and best use cases.\
""",
"youtube_shorts": """\
MAIN IDEA:
(1 sentence: what is the single key point of this short?)

TOPIC:
(What subject or skill does it cover?)

TAGS:
(Comma-separated keywords)

SUMMARY:
A 2–3 sentence explanation of what this short demonstrates or teaches.\
""",

"youtube_channel": """\
CHANNEL NICHE:
(1–2 sentences: what does this channel cover?)

CONTENT STYLE:
(Tutorial / Commentary / Vlog / Documentary / Other)

TARGET AUDIENCE:
(Who watches this channel?)

TAGS:
(Comma-separated keywords)

SUMMARY:
A short paragraph describing the creator, their focus, and what subscribers can expect.\
""",

"substack_article": """\
MAIN ARGUMENT:
(1–2 sentences: what is the author's central claim?)

KEY POINTS:
- Point 1
- Point 2
- Point 3

AUTHOR'S PERSPECTIVE:
(1 sentence: what viewpoint or bias does the author bring?)

TAGS:
(Comma-separated keywords)

SUMMARY:
A short paragraph (3–5 sentences) covering the argument, evidence, and takeaways.\
""",

"notion_page": """\
MAIN TOPIC:
(1–2 sentences: what is this page about?)

KEY SECTIONS:
- Section or heading 1
- Section or heading 2

IMPORTANT DETAILS:
- Key fact or item 1
- Key fact or item 2

TAGS:
(Comma-separated keywords)

SUMMARY:
A short paragraph covering the page's purpose and main content.\
""",

"pdf_document": """\
DOCUMENT TYPE:
(Report / Paper / Manual / Guide / Contract / Other)

MAIN IDEA:
(1–2 sentences: what is this document about?)

KEY SECTIONS:
- Section or topic 1
- Section or topic 2

IMPORTANT DETAILS:
- Fact, finding, or step 1
- Fact, finding, or step 2

TAGS:
(Comma-separated keywords)

SUMMARY:
A short paragraph (3–5 sentences) covering the document's purpose, content, and key points.\
""",

"instagram_post": """\
MAIN MESSAGE:
(1–2 sentences: what is this post about?)

KEY TAKEAWAY:
(What should the viewer do or know after seeing this?)

CONTENT TYPE:
(Tutorial / Motivation / Product / Lifestyle / News / Other)

TAGS:
(Comma-separated keywords)

SUMMARY:
A short paragraph explaining the post's message and audience.\
""",

"instagram_reel": """\
MAIN MESSAGE:
(1–2 sentences: what is this reel demonstrating or saying?)

KEY STEPS OR POINTS:
- Step or point 1
- Step or point 2

CONTENT TYPE:
(Tutorial / Entertainment / Motivation / Product demo / Other)

TAGS:
(Comma-separated keywords)

SUMMARY:
A short paragraph explaining what the reel covers and what viewers will take away.\
""",

"instagram_profile": """\
CREATOR NICHE:
(1–2 sentences: what does this account post about?)

AUDIENCE:
(Who follows this account?)

CONTENT STYLE:
(Educational / Entertainment / Lifestyle / Business / Other)

TAGS:
(Comma-separated keywords)

SUMMARY:
A short paragraph describing the creator, their content focus, and their reach.\
""",

"github_user": """\
DEVELOPER PROFILE:
(1–2 sentences: who is this developer and what do they work on?)

PRIMARY LANGUAGES:
(Languages visible in their repos)

TOP PROJECTS:
- Project 1 — what it does
- Project 2 — what it does

TAGS:
(Comma-separated: e.g. Python, open-source, ML, backend)

SUMMARY:
A short paragraph describing the developer's background, skills, and notable work.\
""",

"loom_video": """\
MAIN TOPIC:
(1–2 sentences: what is being demonstrated or explained?)

KEY STEPS:
- Step or point 1
- Step or point 2
- Step or point 3

AUDIENCE:
(Who is this recording for?)

SUMMARY:
A short paragraph covering what the recording shows and what viewers will learn.\
""",

"plain_text": """\
MAIN IDEA:
(1–2 sentences: what is this note or text about?)

KEY POINTS:
- Point 1
- Point 2
- Point 3

TAGS:
(Comma-separated keywords)

SUMMARY:
A short paragraph summarising the content and any actionable information.\
""",

"json_data": """\
DATA TYPE:
(What kind of data is this? e.g. config, API response, dataset record)

STRUCTURE:
(Top-level keys and what they represent)

KEY VALUES:
- Notable field 1: what it contains
- Notable field 2: what it contains

SUMMARY:
A short paragraph explaining what this JSON represents and how it might be used.\
""",

"markdown": """\
DOCUMENT PURPOSE:
(1–2 sentences: what is this markdown document for?)

MAIN SECTIONS:
- Heading or section 1
- Heading or section 2

KEY CONTENT:
- Important point or fact 1
- Important point or fact 2

TAGS:
(Comma-separated keywords)

SUMMARY:
A short paragraph covering the document's structure and main content.\
""",

"news_headline": """\
EVENT:
(1–2 sentences: what happened?)

KEY PARTIES:
- Who is involved?

WHEN / WHERE:
(Date and location if available)

SIGNIFICANCE:
(1–2 sentences: why does this matter?)

TAGS:
(Comma-separated: e.g. AI, India, startup, policy)

SUMMARY:
A short paragraph explaining the news event and its broader context.\
""",

"huggingface_dataset": """\
MAIN IDEA:
(1–2 sentences: what data does this dataset contain?)

DATA FORMAT:
(Text / Image / Audio / Tabular / Multimodal)

USE CASES:
- Task or use case 1
- Task or use case 2

SIZE & COVERAGE:
(Number of samples, languages, or domains if available)

TAGS:
(Comma-separated: e.g. NLP, classification, multilingual)

SUMMARY:
A short paragraph explaining the dataset's content, format, and ideal use cases.\
""",

"huggingface_space": """\
MAIN IDEA:
(1–2 sentences: what does this Space demo do?)

TASK:
(What ML task does it demonstrate?)

HOW TO USE:
- Input: what the user provides
- Output: what the Space returns

TAGS:
(Comma-separated: e.g. image generation, NLP, demo)

SUMMARY:
A short paragraph explaining what this Space shows and how to interact with it.\
""",

"vimeo_video": """\
MAIN IDEA:
(1–2 sentences: what is this video about?)

KEY THEMES:
- Theme or topic 1
- Theme or topic 2

AUDIENCE:
(Who is this video for?)

TAGS:
(Comma-separated keywords)

SUMMARY:
A short paragraph explaining the video's content and creative intent.\
""",

}

# Default format — used for all source types not in _SOURCE_OUTPUT_FORMAT
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
# STAGE 4 — FINAL SUMMARY PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def build_summary_prompt(data: dict) -> str:
    """
    Build the final LLM prompt from a cleaned + enriched processor output dict.

    - Reads fields in priority order (_PRIORITY_FIELDS), then any extras.
    - Skips pure metadata (_META_FIELDS) — these are routing info, not content.
    - Picks the correct task instruction and output format by source_type.
    - Handles list[str], list[dict], str, and bool fields uniformly.

    Called by: llm/summarizer.py → summarize_data()
    Receives:  dict already cleaned by utils/cleaner.py
               and enriched by llm/pipeline.py Stage 3
    """
    if not data:
        return "No content was provided. Please try a different resource."

    source_type   = data.get("source_type", "")
    instruction   = _SOURCE_INSTRUCTIONS.get(source_type, _DEFAULT_INSTRUCTION)
    output_format = _SOURCE_OUTPUT_FORMAT.get(source_type, _DEFAULT_OUTPUT_FORMAT)

    # Special case: GitHub repo with no README — stricter instruction
    if source_type == "github_repo" and not data.get("has_readme", True):
        instruction = (
            "This GitHub repository has NO README file. "
            "Summarize STRICTLY from the source files provided below. "
            "Do NOT invent features, tech stack, or descriptions that are not "
            "explicitly visible in the files. If something is unclear, say so."
        )

    content_parts: list[str] = []
    seen: set[str] = set()

    # Priority fields first, then any extra content fields not already listed
    all_keys = _PRIORITY_FIELDS + [
        k for k in data
        if k not in _PRIORITY_FIELDS
        and k not in _META_FIELDS
        and k != "source_type"
    ]

    for key in all_keys:
        if key in seen or key not in data:
            continue
        if key in _META_FIELDS:
            continue
        seen.add(key)

        value = data[key]
        label = _FIELD_LABELS.get(key, key.replace("_", " ").title())

        # ── Skip empty values — but keep False booleans ───────────────────
        if value is None or value == "" or value == []:
            continue

        # ── list[dict] — e.g. top_repos, structured comment objects ──────
        if isinstance(value, list) and value and isinstance(value[0], dict):
            rendered = _render_list_of_dicts(value)
            if rendered:
                content_parts.append(f"{label}:\n{rendered}")

        # ── list[str] or list[int] — e.g. top_comments, tags ─────────────
        elif isinstance(value, list):
            items = "\n".join(f"- {c}" for c in value[:20] if c)
            if items:
                content_parts.append(f"{label}:\n{items}")

        # ── str ───────────────────────────────────────────────────────────
        elif isinstance(value, str) and value.strip():
            content_parts.append(f"{label}:\n{value.strip()}")

        # ── bool — e.g. is_verified, is_business ─────────────────────────
        elif isinstance(value, bool):
            content_parts.append(f"{label}: {'Yes' if value else 'No'}")

        # ── int / float — only include if non-zero and meaningful ─────────
        elif isinstance(value, (int, float)) and value:
            content_parts.append(f"{label}: {value}")

    if not content_parts:
        return (
            "The content could not be extracted or was empty. "
            "Please try a different resource or paste the content directly."
        )

    content = "\n\n".join(content_parts)

    return f"""Task: {instruction}

Rules: extract stated facts only — no filler, ads, or invented content.
Mention real tool names, steps, and technologies. Fill ALL format sections.
Do NOT write intro phrases like "Here is the summary" or "Based on the content".
Do NOT leave any section blank — write "Not available" if truly missing.
Do NOT invent information that is not present in the content.
Do NOT truncate the SUMMARY section — always write the full paragraph.

Content:
{content}

Return output in this exact format:

{output_format}

MAIN IDEA:
"""


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — MERGE PROMPT (multi-chunk)
# ─────────────────────────────────────────────────────────────────────────────

def build_merge_prompt(partial_summaries: list[str], source_type: str = "") -> str:
    """
    Build a prompt to merge multiple chunk summaries into one final answer.
    Called by summarizer.py when content was too long for one context window.

    Called by: llm/summarizer.py → summarize_data()
    """
    # ── Fix: use source-specific format, not always the generic default ──
    output_format = _SOURCE_OUTPUT_FORMAT.get(source_type, _DEFAULT_OUTPUT_FORMAT)

    combined = "\n\n---\n\n".join(
        f"[Part {i + 1}]\n{s.strip()}" for i, s in enumerate(partial_summaries)
    )

    return f"""The following are partial summaries of different sections of the same resource.
Combine them into one single, clean, non-repetitive structured summary.

Rules: do not mention that content was split. Fill ALL sections. No intro phrases.
Do NOT invent information not present in the summaries below.
Do NOT leave any section blank — write "Not available" if truly missing.

Partial Summaries:
{combined}

Return the final output in this exact format:

{output_format}

MAIN IDEA:
"""



# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — CLASSIFIER PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def build_classifier_prompt(user_input: str) -> str:
    """
    Build a prompt to classify what type of content the user input is.
    Used by llm/llm_classifier.py (Stage 1 of the pipeline).

    Returns a prompt that expects a raw JSON response.

    Called by: llm/pipeline.py → classify()
    """
    return f"""You are a content type classifier for a developer knowledge agent.
Classify the input below and return ONLY a valid JSON object. No explanation. No markdown.

Input:
{user_input[:800]}

Return exactly this JSON structure:
{{
  "source_type": "<type>",
  "confidence": "<high|medium|low>",
  "reason": "<one short sentence>"
}}

Valid source_type values:
youtube_video, youtube_shorts, youtube_playlist, youtube_channel,
github_repo, github_user, github_file, github_gist,
web, medium_article, substack_article, notion_page, arxiv_paper,
reddit_post, reddit_subreddit, instagram_post, instagram_reel,
instagram_profile, linkedin_profile, linkedin_post, linkedin_company,
huggingface_model, huggingface_dataset, huggingface_space,
loom_video, vimeo_video, local_image, pdf_document,
code_snippet, notebook, plain_text, news_headline, json_data, markdown,
unsupported
"""


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — EXTRACTION GUIDANCE PROMPT
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — PER-TYPE GUIDANCE EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────

_GUIDANCE_EXAMPLES: dict[str, str] = {

    "github_repo": """{
  "focus_on": ["source files", "package.json", "pyproject.toml", "languages used", "project purpose"],
  "skip":     ["test files", "CI config", "LICENSE text", "build artifacts", "lock files"],
  "infer":    ["target audience", "maturity level", "similar tools", "use case"],
  "context":  "This is a GitHub repository. Summarize from whatever files are available."
}""",

    "github_file": """{
  "focus_on": ["function definitions", "class definitions", "imports", "docstrings", "logic flow"],
  "skip":     ["comments about formatting", "auto-generated code", "boilerplate"],
  "infer":    ["programming language", "role in larger project", "dependencies"],
  "context":  "This is a single source code file from a GitHub repository."
}""",

    "github_gist": """{
  "focus_on": ["core logic", "function purpose", "input/output", "language used"],
  "skip":     ["unrelated comments", "version history notes"],
  "infer":    ["use case", "language", "when to apply this pattern"],
  "context":  "This is a GitHub Gist — a standalone code snippet or note."
}""",

    "github_user": """{
  "focus_on": ["bio", "top repositories", "primary languages", "notable projects"],
  "skip":     ["forked repos", "archived repos", "follower counts"],
  "infer":    ["developer specialisation", "experience level", "open source contributions"],
  "context":  "This is a GitHub user profile page."
}""",

    "youtube_video": """{
  "focus_on": ["transcript", "key concepts explained", "tools and technologies mentioned", "steps or tutorial flow"],
  "skip":     ["sponsor segments", "like/subscribe calls", "filler phrases", "channel intros"],
  "infer":    ["target audience", "difficulty level", "related tools or alternatives"],
  "context":  "This is a YouTube educational or tutorial video."
}""",

    "youtube_shorts": """{
  "focus_on": ["single key message", "core tip or trick", "topic demonstrated"],
  "skip":     ["filler", "hashtags", "subscribe prompts"],
  "infer":    ["topic category", "target audience"],
  "context":  "This is a YouTube Short — a video under 60 seconds with one core point."
}""",

    "youtube_playlist": """{
  "focus_on": ["playlist title", "series theme", "topics covered across videos", "progression or curriculum"],
  "skip":     ["individual video details", "view counts", "upload dates"],
  "infer":    ["overall learning path", "target audience", "difficulty progression"],
  "context":  "This is a YouTube playlist — a curated series of related videos."
}""",

    "youtube_channel": """{
  "focus_on": ["channel name", "content niche", "upload style", "target audience"],
  "skip":     ["individual video titles", "subscriber counts"],
  "infer":    ["creator expertise", "content frequency", "similar channels"],
  "context":  "This is a YouTube channel page."
}""",

    "arxiv_paper": """{
  "focus_on": ["abstract", "problem statement", "proposed method", "experiments", "results", "conclusion"],
  "skip":     ["author affiliations", "reference list numbers", "LaTeX formatting artifacts", "acknowledgements"],
  "infer":    ["difficulty level", "real-world applications", "related prior work", "limitations"],
  "context":  "This is an academic research paper from ArXiv."
}""",

    "pdf_document": """{
  "focus_on": ["document purpose", "main sections", "key findings or steps", "conclusions"],
  "skip":     ["page numbers", "headers/footers", "table of contents", "bibliography"],
  "infer":    ["document type", "intended audience", "publication date"],
  "context":  "This is text extracted from a PDF document."
}""",

    "web": """{
  "focus_on": ["article body", "main argument", "key facts", "conclusions", "author claims"],
  "skip":     ["cookie banners", "navigation menus", "ads", "footer links", "related article links"],
  "infer":    ["author perspective", "publication date", "target audience", "article bias"],
  "context":  "This is a web article or blog post."
}""",

    "medium_article": """{
  "focus_on": ["main argument", "supporting points", "examples used", "practical takeaways"],
  "skip":     ["clap/follow prompts", "author bio boilerplate", "related article recommendations"],
  "infer":    ["author background", "target audience", "difficulty level"],
  "context":  "This is a Medium article — typically opinion, tutorial, or analysis."
}""",

    "substack_article": """{
  "focus_on": ["author's central claim", "key arguments", "evidence cited", "calls to action"],
  "skip":     ["subscription prompts", "sponsor mentions", "share buttons text"],
  "infer":    ["author's viewpoint or bias", "newsletter niche", "target reader"],
  "context":  "This is a Substack newsletter post."
}""",

    "notion_page": """{
  "focus_on": ["headings", "bullet points", "structured content", "main topic"],
  "skip":     ["empty sections", "template placeholder text", "navigation blocks"],
  "infer":    ["document purpose", "intended audience", "project or team context"],
  "context":  "This is a Notion page or document."
}""",

    "reddit_post": """{
  "focus_on": ["original post text", "top comments", "consensus opinions", "key debate points", "upvoted answers"],
  "skip":     ["low-effort replies", "mod notes", "off-topic tangents", "deleted comments"],
  "infer":    ["community sentiment", "best answer", "related subreddits"],
  "context":  "This is a Reddit post or discussion thread."
}""",

    "reddit_subreddit": """{
  "focus_on": ["subreddit description", "rules", "common post themes", "community purpose"],
  "skip":     ["moderator usernames", "sidebar widget text", "post flair lists"],
  "infer":    ["community size and activity", "typical user profile", "related communities"],
  "context":  "This is a Reddit community (subreddit) page."
}""",

    "instagram_post": """{
  "focus_on": ["caption text", "core message", "call to action", "top comments"],
  "skip":     ["hashtag spam", "emoji-only comments", "spam replies", "follow-for-follow comments"],
  "infer":    ["target audience", "content category", "brand or creator niche"],
  "context":  "This is an Instagram photo or carousel post caption."
}""",

    "instagram_reel": """{
  "focus_on": ["caption", "key message demonstrated", "steps or tips shown", "top comments"],
  "skip":     ["hashtag spam", "irrelevant emoji comments", "promotional spam"],
  "infer":    ["content category", "target audience", "creator niche"],
  "context":  "This is an Instagram Reel — a short-form video post."
}""",

    "instagram_profile": """{
  "focus_on": ["bio", "content niche", "recent post captions", "follower count context"],
  "skip":     ["follower/following exact numbers", "post counts"],
  "infer":    ["content style", "target audience", "monetisation or brand type"],
  "context":  "This is an Instagram creator or brand profile."
}""",

    "huggingface_model": """{
  "focus_on": ["model description", "task type", "usage instructions", "training data", "limitations"],
  "skip":     ["raw config JSON", "eval metric tables", "contributor lists"],
  "infer":    ["best use cases", "comparable models", "required hardware"],
  "context":  "This is a HuggingFace model card."
}""",

    "huggingface_dataset": """{
  "focus_on": ["dataset description", "data format", "size", "languages", "use cases", "license"],
  "skip":     ["raw schema definitions", "contributor lists", "version changelogs"],
  "infer":    ["best ML tasks for this data", "data quality", "coverage gaps"],
  "context":  "This is a HuggingFace dataset card."
}""",

    "huggingface_space": """{
  "focus_on": ["what the demo does", "input type", "output type", "model used", "how to interact"],
  "skip":     ["technical deployment config", "Dockerfile contents"],
  "infer":    ["target users", "practical applications", "underlying model"],
  "context":  "This is a HuggingFace Space — an interactive ML demo app."
}""",

    "loom_video": """{
  "focus_on": ["main topic demonstrated", "key steps shown", "tools used", "context of recording"],
  "skip":     ["filler phrases", "mouse movement commentary", "off-topic tangents"],
  "infer":    ["intended audience", "project or workflow context", "follow-up actions"],
  "context":  "This is a Loom screen recording, typically a walkthrough or explanation."
}""",

    "vimeo_video": """{
  "focus_on": ["video description", "main theme", "key visuals or topics described", "creator intent"],
  "skip":     ["platform UI text", "related video titles"],
  "infer":    ["target audience", "creative or educational intent", "genre"],
  "context":  "This is a Vimeo video."
}""",

    "local_image": """{
  "focus_on": ["all readable text", "UI elements", "code visible", "diagrams", "data shown"],
  "skip":     ["image metadata", "file path info"],
  "infer":    ["what the image is showing", "context of the screenshot", "tool or app visible"],
  "context":  "This is OCR-extracted text from a local image or screenshot."
}""",

    "plain_text": """{
  "focus_on": ["main topic", "key facts", "actionable points", "conclusions"],
  "skip":     ["filler sentences", "repeated ideas"],
  "infer":    ["purpose of this note", "intended audience", "related topics"],
  "context":  "This is plain text pasted directly by the user — could be notes, an article excerpt, or a message."
}""",

    "code_snippet": """{
  "focus_on": ["function purpose", "input/output", "key logic steps", "language used", "dependencies"],
  "skip":     ["commented-out dead code", "formatting-only comments"],
  "infer":    ["use case", "language", "where this fits in a larger system"],
  "context":  "This is a standalone code snippet pasted by the user."
}""",

    "json_data": """{
  "focus_on": ["top-level keys", "nested structure", "value types", "what the data represents"],
  "skip":     ["deeply nested repeated values", "null fields"],
  "infer":    ["data source or API origin", "intended use", "data schema"],
  "context":  "This is a JSON data blob pasted by the user."
}""",

    "markdown": """{
  "focus_on": ["headings", "bullet points", "code blocks", "links", "bold text"],
  "skip":     ["empty sections", "placeholder text"],
  "infer":    ["document purpose", "target audience", "project context"],
  "context":  "This is a markdown-formatted document."
}""",

    "news_headline": """{
  "focus_on": ["event described", "key parties", "date and location", "significance"],
  "skip":     ["ads", "related article links", "author bios"],
  "infer":    ["broader impact", "related events", "industry affected"],
  "context":  "This is a news headline or short news snippet."
}""",
}

# Fallback example for any unknown type
_DEFAULT_GUIDANCE_EXAMPLE = """{
  "focus_on": ["main content", "key facts", "important details", "conclusions"],
  "skip":     ["ads", "navigation", "boilerplate", "repeated content"],
  "infer":    ["target audience", "difficulty level", "related topics"],
  "context":  "This is a web resource. Extract the most important information."
}"""


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — EXTRACTION GUIDANCE PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def build_guidance_prompt(user_input: str, source_type: str) -> str:
    """
    Build a prompt that tells the processor what to focus on and skip.
    Used by llm/pipeline.py Stage 2 (extract_guidance).

    Returns a prompt that expects a raw JSON response.

    Called by: llm/pipeline.py → extract_guidance()
    """
    example = _GUIDANCE_EXAMPLES.get(source_type, _DEFAULT_GUIDANCE_EXAMPLE)

    return f"""You are helping a developer knowledge agent decide what to extract from content.

Source type : {source_type}
Input       : {user_input[:600]}

Return ONLY a valid JSON object. No explanation. No markdown.

{{
  "focus_on": ["list of fields or sections that matter most"],
  "skip":     ["list of things to ignore or filter out"],
  "infer":    ["things that can be inferred even if not explicit"],
  "context":  "one sentence of background context for this content type"
}}

Example for {source_type}:
{example}
"""


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — ENRICHMENT PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def build_enrich_prompt(cleaned_data: dict, guidance: dict) -> str:
    """
    Build a prompt that fills gaps in cleaned processor output.
    Used by llm/pipeline.py Stage 3 (enrich).

    Returns a prompt that expects a raw JSON response of new/improved fields.

    Called by: llm/pipeline.py → enrich()
    """
    # Truncate each field value to avoid overflowing the context window
    have = {
        k: (str(v)[:300] if isinstance(v, str) else v)
        for k, v in cleaned_data.items()
        if v and k not in ("source_type", "url")
    }

    source_type = cleaned_data.get("source_type", "")
    infer_list  = guidance.get("infer", [])
    context_str = guidance.get("context", "")

    return f"""You are enriching structured content data for a knowledge extraction agent.

Source type: {source_type}

What we already have:
{json.dumps(have, indent=2, ensure_ascii=False)[:1500]}

Background context:
{context_str}

Things we should try to infer:
{json.dumps(infer_list)}

Task: Add any missing but clearly inferable fields. Return ONLY a valid JSON object
with NEW fields to merge into the data. Only include fields you are confident about.
Do not repeat or rewrite fields that already exist and have good values.

Valid new fields you may add (pick only what applies):

  AUDIENCE & DIFFICULTY
  ─────────────────────
  inferred_audience      — who this content is for
                           e.g. "backend Python developers", "ML researchers", "beginners"

  inferred_difficulty    — one of: Beginner / Intermediate / Advanced

  inferred_prerequisites — what the reader/viewer needs to know first
                           e.g. "basic Python knowledge", "familiarity with transformers"

  CATEGORISATION
  ──────────────
  inferred_category      — a short content category label
                           e.g. "DevOps tool", "ML research paper", "tutorial", "news"

  inferred_use_case      — 1 sentence: what someone would use this for
                           e.g. "Building REST APIs in Python with automatic docs"

  RELATIONSHIPS
  ─────────────
  related_tools          — comma-separated similar tools, libraries, or alternatives
                           e.g. "Flask, Django, Starlette" (for a FastAPI repo)

  key_entities           — comma-separated notable people, organisations, or products mentioned
                           e.g. "OpenAI, Sam Altman, GPT-4"

  GAPS
  ────
  missing_context        — 1 sentence of important context NOT present in the content
                           e.g. "This paper builds on BERT, published in 2018 by Google."

Return empty {{}} if nothing meaningful can be inferred.

Example output for a github_repo:
{{
  "inferred_audience":      "Python backend developers building REST APIs",
  "inferred_difficulty":    "Intermediate",
  "inferred_prerequisites": "Basic Python and HTTP knowledge",
  "inferred_category":      "Web framework",
  "inferred_use_case":      "Building production-ready REST APIs with automatic OpenAPI documentation",
  "related_tools":          "Flask, Django, Starlette, Express.js",
  "missing_context":        "FastAPI is built on top of Starlette and Pydantic."
}}
"""
