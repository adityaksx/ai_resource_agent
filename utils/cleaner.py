import re


# -------------------------
# Basic text cleaning
# -------------------------

def clean_text(text: str) -> str:

    if not text:
        return ""

    # remove urls
    text = re.sub(r"http\S+", "", text)

    # remove hashtags
    text = re.sub(r"#\w+", "", text)

    # remove emojis / symbols
    text = re.sub(r"[^\w\s.,!?]", "", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -------------------------
# Remove duplicate lines
# -------------------------

def remove_duplicates(lines):

    seen = set()
    result = []

    for line in lines:

        key = line.lower()

        if key not in seen:
            seen.add(key)
            result.append(line)

    return result


# -------------------------
# Split into sentences
# -------------------------

def split_sentences(text):

    sentences = re.split(r"[.!?]\s+", text)

    return [s.strip() for s in sentences if len(s.strip()) > 20]


# -------------------------
# Compress content
# -------------------------

def compress_text(text, max_sentences=50):

    sentences = split_sentences(text)

    sentences = remove_duplicates(sentences)

    return " ".join(sentences[:max_sentences])


# -------------------------
# Clean comments
# -------------------------

def clean_comments(comments, max_comments=30):

    cleaned = []

    for c in comments:

        c = clean_text(c)

        if len(c) > 10:
            cleaned.append(c)

    cleaned = remove_duplicates(cleaned)

    return cleaned[:max_comments]


# -------------------------
# Main cleaner
# -------------------------

def clean_processor_output(data):

    """
    Takes output from processors and prepares compact input for LLM
    """

    result = {}

    if "content" in data:
        result["content"] = compress_text(clean_text(data["content"]))

    if "transcript" in data:
        result["transcript"] = compress_text(clean_text(data["transcript"]))

    if "caption" in data:
        result["caption"] = clean_text(data["caption"])

    if "overview" in data:
        result["overview"] = compress_text(clean_text(data["overview"]))

    if "description" in data:
        result["description"] = clean_text(data["description"])

    if "unique_comments" in data:
        result["comments"] = clean_comments(data["unique_comments"])

    return result