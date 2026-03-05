def build_summary_prompt(data: dict) -> str:
    """
    Build an optimized prompt for small LLMs (7B–8B).
    """

    content_parts = []

    if "title" in data:
        content_parts.append(f"Title:\n{data['title']}")

    if "description" in data:
        content_parts.append(f"Description:\n{data['description']}")

    if "caption" in data:
        content_parts.append(f"Caption:\n{data['caption']}")

    if "overview" in data:
        content_parts.append(f"Project Overview:\n{data['overview']}")

    if "content" in data:
        content_parts.append(f"Article Content:\n{data['content']}")

    if "transcript" in data:
        content_parts.append(f"Video Transcript:\n{data['transcript']}")

    if "comments" in data and isinstance(data["comments"], list):
        comments = "\n".join(data["comments"][:20])
        content_parts.append(f"User Comments:\n{comments}")

    content = "\n\n".join(content_parts)

    prompt = f"""
You are an intelligent knowledge extraction assistant.

Your task is to understand the content and extract the most important ideas.

Instructions:
- Ignore filler words.
- Focus on concepts, insights, and explanations.
- Combine information from transcript, article, and comments.
- Avoid repeating sentences.
- Keep answers concise.

Content:
{content}

Return output in this format:

MAIN IDEA:
(1–2 sentences explaining the core topic)

KEY INSIGHTS:
- Insight 1
- Insight 2
- Insight 3
- Insight 4

IMPORTANT DETAILS:
- technical ideas
- tools mentioned
- methods or steps

SUMMARY:
A short explanation (5–8 sentences) of what the content teaches.
"""

    return prompt