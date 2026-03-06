def build_vault_metadata(cleaned_data: dict):
    source = cleaned_data.get("source_type", "")

    title = cleaned_data.get("title")
    snippet = ""

    if source.startswith("github"):
        title = cleaned_data.get("repo") or title
        snippet = cleaned_data.get("description", "")

    elif source.startswith("youtube"):
        title = cleaned_data.get("title")
        snippet = cleaned_data.get("description", "")

    elif source == "local_image":
        title = cleaned_data.get("filename")
        snippet = cleaned_data.get("ocr_text", "")[:150]

    elif source == "plain_text":
        text = cleaned_data.get("text", "")
        title = text[:60]
        snippet = text[:160]

    else:
        snippet = str(cleaned_data)[:160]

    return {
        "vault_title": title,
        "vault_snippet": snippet
    }