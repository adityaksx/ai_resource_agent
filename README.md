```markdown
# 🤖 AI Resource Agent

A fully async, multi-source AI agent that accepts URLs, plain text, or images — automatically detects the source type, routes to the right processor, runs a clean → enrich → summarize LLM pipeline, and saves everything to a local database. Served via a FastAPI web interface.

---

## ✨ Features

- **Multi-Source Input** — Paste a YouTube link, GitHub repo, Instagram post, web article, ArXiv paper, Hugging Face model, plain text, or a local image/file
- **Smart Source Detection** — Two-stage detection: fast regex rules first, then an LLM classifier fallback for ambiguous inputs
- **Async Pipeline** — Fully `async/await` throughout; no event-loop blocking
- **Multi-Stage LLM Pipeline** — classify → extract guidance → clean → enrich → summarize
- **Persistent Database** — Every processed resource is saved with vault metadata (title, snippet, source type, status)
- **FastAPI Web UI** — Chat-style interface accessible from the browser
- **CLI Mode** — Run `python main.py` for a terminal REPL

---

## 📁 Project Structure

```
ai_resource_agent/
│
├── main.py                  # Central async router & entry point
├── config.py                # Configuration (env vars, constants)
├── .env                     # API keys / secrets (not committed)
│
├── processors/
│   ├── youtube_processor.py    # YouTube videos, shorts, playlists
│   ├── github_processor.py     # GitHub repos, files, gists
│   ├── web_processor.py        # General web, Medium, Substack, ArXiv, HuggingFace, Reddit, etc.
│   ├── instagram_processor.py  # Instagram posts & reels
│   ├── text_processor.py       # Plain text / pasted content
│   └── image_processor.py      # Local images with OCR
│
├── llm/
│   ├── pipeline.py             # classify(), extract_guidance(), enrich()
│   ├── summarizer.py           # summarize_data(), call_llm()
│   ├── prompt_builder.py       # Source-specific prompt construction
│   ├── llm_classifier.py       # LLM-based source type classifier
│   ├── ollama_client.py        # Ollama local LLM client
│   └── embeddings.py           # Embedding utilities
│
├── utils/
│   ├── source_detector.py      # Regex-based source detection (Stage 1)
│   └── cleaner.py              # Cleans & normalises processor output dicts
│
├── database/
│   └── db.py                   # SQLite init, save_resource(), queries
│
└── web/
    ├── app.py                  # FastAPI app — /chat endpoint + static serving
    ├── templates/              # Jinja2 HTML templates
    └── static/                 # CSS / JS assets
```

---

## 🔄 How It Works

```
User Input (URL / text / image)
        │
        ▼
  Stage 1: Rule-based source detection  (utils/source_detector.py)
        │
        ▼ (ambiguous? → Stage 2)
  Stage 2: LLM classifier fallback      (llm/llm_classifier.py)
        │
        ▼
  Route to Processor
  ┌─────────────────────────────────────────────┐
  │  YouTube · GitHub · Web · Instagram         │
  │  Text · Image (OCR)                         │
  └─────────────────────────────────────────────┘
        │
        ▼
  Stage 3: extract_guidance()  →  clean()  →  enrich()
        │
        ▼
  Stage 4: summarize_data()  via  call_llm()
        │
        ▼
  Save to SQLite DB  +  Return LLM output to user
```

---

## 🌐 Supported Sources

| Category | Sources |
|---|---|
| **Video** | YouTube videos, shorts, playlists |
| **Code** | GitHub repos, files, gists |
| **Social** | Instagram posts, reels, Reddit posts/subreddits |
| **Articles** | Medium, Substack, Notion pages, ArXiv papers |
| **AI/ML** | HuggingFace models, datasets, spaces |
| **Web** | Any general webpage, Pastebin, Loom, Vimeo |
| **Files** | PDF (URL or local), images, plain text, code files, notebooks |
| **Text** | Pasted plain text or notes |

> ⚠️ **Not supported:** LinkedIn profiles/company pages (login-protected). Paste the content text directly instead.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally (or configure another LLM backend)

### Installation

```bash
git clone https://github.com/adityaksx/ai_resource_agent.git
cd ai_resource_agent
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
OLLAMA_MODEL=llama3
# add any other keys required by processors
```

### Run (Web UI)

```bash
uvicorn web.app:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

### Run (CLI)

```bash
python main.py
```

Paste any URL, text, or a local file/image path at the prompt. Type `exit` to quit.

---

## 🧩 Key Modules

### `main.py`
The central async router. Accepts any combination of URLs + text + image paths, detects source types, delegates to the right processor, runs the full pipeline, saves to DB, and returns the final LLM output.

### `llm/pipeline.py`
Implements the multi-stage async pipeline:
- `classify(input)` — identifies the content type
- `extract_guidance(input, source_type)` — pulls focus hints for the summarizer
- `enrich(cleaned_data, guidance)` — augments data with context before summarization

### `llm/prompt_builder.py`
Constructs source-specific prompts. Each source type (YouTube, GitHub, ArXiv, etc.) gets a tailored prompt structure for the best LLM output.

### `processors/`
Each processor extracts raw structured data from its source and returns a normalized dict. The `web_processor.py` handles the broadest range of URLs including ArXiv, HuggingFace, Reddit, and generic sites.

### `database/db.py`
SQLite-backed storage. Saves every resource with `vault_title`, `vault_snippet`, `source`, `status`, `raw_input`, `raw_data`, `cleaned_data`, and `llm_output`.

---

## 🛠️ Architecture Notes

- **No blocking I/O** — all network calls and LLM calls use `async/await`
- **Separation of concerns** — routing, prompting, cleaning, and DB operations are in separate modules
- **Graceful error handling** — friendly messages for JS-only sites, 404s, timeouts, and login-protected pages
- **Mixed input** — sending multiple URLs + text in one request is supported; the agent processes each independently then synthesizes a combined insight

---

## 📄 License

This project is open source. See [LICENSE](LICENSE) for details.
```
