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
    "ocr_text",           # image OCR — before generic content
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
    # ── Pipeline Stage 3 enriched fields ─────────────────────────────────
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
    "readme":              "README",
    "code":                "Code",
    "comments":            "User Comments",
    "unique_comments":     "User Comments",
    "summary":             "Summary",
    # Pipeline enriched fields
    "inferred_audience":   "Target Audience (inferred)",
    "inferred_difficulty": "Difficulty Level (inferred)",
    "related_tools":       "Related Tools / Alternatives",
    "missing_context":     "Additional Context",
}

# Pure metadata — never shown in the content block, only used for routing.
# These are preserved by cleaner.py but excluded here from the prompt body.
_META_FIELDS: set[str] = {
    "source_type", "url", "video_id", "channel",
    "author", "date", "source",
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
    "github_gist":      "This is a GitHub Gist. Explain what the code or note does and when you would use it.",
    "json_data":        "This is structured JSON data. Describe what it represents and what each key section contains.",
    "markdown":         "This is a markdown document. Extract the main topics, structure, and important information.",
}

_DEFAULT_INSTRUCTION = "Understand the content and extract the most important ideas, facts, and insights."


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
    - Handles list fields (comments) and string fields uniformly.

    Called by: llm/summarizer.py → summarize_data()
    Receives:  dict already cleaned by utils/cleaner.py
               and enriched by llm/pipeline.py Stage 3
    """
    if not data:
        return "No content was provided. Please try a different resource."

    source_type   = data.get("source_type", "")
    instruction   = _SOURCE_INSTRUCTIONS.get(source_type, _DEFAULT_INSTRUCTION)
    output_format = _SOURCE_OUTPUT_FORMAT.get(source_type, _DEFAULT_OUTPUT_FORMAT)

    if source_type == "github_repo" and not data.get("has_readme", True):
        instruction = (
            "This GitHub repository has NO README file. "
            "Summarize STRICTLY from the source files provided below. "
            "Do NOT invent features, tech stack, or descriptions that are not "
            "explicitly visible in the files. If something is unclear, say so."
        )

    content_parts: list[str] = []
    seen: set[str] = set()

    # Priority fields first, then any extra fields not in the priority list
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
            "Please try a different resource or paste the content directly."
        )

    content = "\n\n".join(content_parts)

    return f"""You are an intelligent knowledge extraction assistant.

Task: {instruction}

Instructions:
- Focus on concepts, facts, insights, and explanations.
- Ignore filler words, ads, sponsors, and navigation text.
- If the content is OCR from a screenshot, reconstruct what it was showing.
- If the content is code, explain what it does clearly.
- Avoid repeating sentences or restating the same idea.
- Be specific — mention actual tools, technologies, names, and steps.
- Keep answers concise and useful.
- Fill in ALL sections of the output format below.

Content:
{content}

Return output in this exact format:

{output_format}
"""


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — MERGE PROMPT (multi-chunk)
# ─────────────────────────────────────────────────────────────────────────────

def build_merge_prompt(partial_summaries: list[str]) -> str:
    """
    Build a prompt to merge multiple chunk summaries into one final answer.
    Called by summarizer.py when content was too long for one context window.

    Called by: llm/summarizer.py → summarize_data()
    """
    combined = "\n\n---\n\n".join(
        f"[Part {i + 1}]\n{s.strip()}" for i, s in enumerate(partial_summaries)
    )

    return f"""You are an intelligent knowledge extraction assistant.

The following are partial summaries of different sections of the same resource.
Combine them into one single, clean, non-repetitive structured summary.
Do not mention that the content was split into parts.

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

def build_guidance_prompt(user_input: str, source_type: str) -> str:
    """
    Build a prompt that tells the processor what to focus on and skip.
    Used by llm/pipeline.py Stage 2 (extract_guidance).

    Returns a prompt that expects a raw JSON response.

    Called by: llm/pipeline.py → extract_guidance()
    """
    if source_type == "github_repo":
        example = """{
  "focus_on": ["source files", "package.json", "languages used", "project purpose"],
  "skip":     ["test files", "CI config", "LICENSE text", "build artifacts"],
  "infer":    ["target audience", "maturity level", "similar tools"],
  "context":  "This is a GitHub repository. Summarize from whatever files are available."
}"""
    else:
        example = """{
  "focus_on": ["README", "tech stack", "project purpose", "installation steps"],
  "skip":     ["test files", "CI config", "CHANGELOG", "LICENSE text"],
  "infer":    ["target audience", "maturity level", "similar tools"],
  "context":  "This is an open source project hosted on GitHub."
}"""

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

Example for github_repo:
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

    infer_list  = guidance.get("infer", [])
    context_str = guidance.get("context", "")

    return f"""You are enriching structured content data for a knowledge extraction agent.

What we already have:
{json.dumps(have, indent=2, ensure_ascii=False)[:1500]}

Background context:
{context_str}

Things we should try to infer:
{json.dumps(infer_list)}

Task: Add any missing but clearly inferable fields. Return ONLY a valid JSON object
with NEW fields to merge into the data. Only include fields you are confident about.
Do not repeat or rewrite fields that already exist and have good values.

Valid new fields you may add:
  inferred_audience    — who this content is for (e.g. "backend Python developers")
  inferred_difficulty  — Beginner / Intermediate / Advanced
  related_tools        — comma-separated similar tools or alternatives
  missing_context      — one sentence of context not present in the content

Return empty {{}} if nothing meaningful can be inferred.
"""
