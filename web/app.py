"""
web/app.py
----------
FastAPI application for the AI Resource Agent.

Handles:
  - GET  /                     → serve chat UI
  - GET  /resources            → serve Media Vault UI
  - POST /chat                 → process message (text + URLs) and uploaded images
  - GET  /api/resources        → list all saved resources (JSON)
  - GET  /api/resources/{id}   → get single resource detail (JSON)
  - DELETE /api/resources/{id} → delete a resource
"""

import os
import sys
import shutil
import time
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR   = Path(__file__).parent
ROOT_DIR   = BASE_DIR.parent
IMAGES_DIR = ROOT_DIR / "storage" / "images"
PDFS_DIR   = ROOT_DIR / "storage" / "pdfs"

sys.path.insert(0, str(ROOT_DIR))

from main        import process_link, process_text_input, process_image_input
from database.db import init_db, get_resources, get_resource, delete_resource


app = FastAPI(title="AI Resource Agent")

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)


@app.on_event("startup")
async def startup():
    init_db()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/storage/images",
        StaticFiles(directory=str(IMAGES_DIR)),
        name="images",
    )


# ── HTML routes ───────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home():
    return (BASE_DIR / "templates" / "chat.html").read_text(encoding="utf-8")


@app.get("/resources", response_class=HTMLResponse)
def resources_page():
    return (BASE_DIR / "templates" / "resources.html").read_text(encoding="utf-8")


# ── API: list all resources ───────────────────────────────

@app.get("/api/resources")
def api_list_resources(limit: int = 500):
    rows = get_resources(limit=limit)
    items = []
    for row in rows:
        # columns: 0=id, 1=source, 2=url, 3=title,
        #          4=raw_input, 5=raw_data, 6=cleaned_data, 7=llm_output,
        #          8=files, 9=status, 10=error, 11=created_at,
        #          12=vault_title, 13=vault_snippet
        items.append({
            "id":            row[0],
            "source":        row[1],
            "url":           row[2],
            "title":         row[3],
            "llm_output":    row[7],
            "status":        row[9],
            "created_at":    row[11],
            "vault_title":   row[12],
            "vault_snippet": row[13],
            "session_id":    row[14],
        })
    return {"resources": items}


# ── API: single resource detail ───────────────────────────

@app.get("/api/resources/{resource_id}")

def api_get_resource(resource_id: int):
    row = get_resource(resource_id)
    if not row:
        raise HTTPException(status_code=404, detail="Resource not found")
    return {
        "id":            row[0],
        "source":        row[1],
        "url":           row[2],
        "title":         row[3],
        "raw_input":     row[4],
        "raw_data":      row[5],
        "cleaned_data":  row[6],
        "llm_output":    row[7],
        "files":         row[8],
        "status":        row[9],
        "error":         row[10],
        "created_at":    row[11],
        "vault_title":   row[12],
        "vault_snippet": row[13],
    }


# ── API: delete resource ──────────────────────────────────

@app.delete("/api/resources/{resource_id}")
def api_delete_resource(resource_id: int):
    row = get_resource(resource_id)
    if not row:
        raise HTTPException(status_code=404, detail="Resource not found")
    delete_resource(resource_id)
    return {"ok": True, "deleted_id": resource_id}


# ── Chat ──────────────────────────────────────────────────

@app.post("/chat")
async def chat(
    message:    Optional[str]              = Form(default=None),
    images:     Optional[List[UploadFile]] = File(default=None),
    session_id: Optional[str]              = Form(default=None),   # ← ADD
):
    # Use session ID from frontend (same thread = same ID), fallback to new one
    try:
        sid = int(session_id) if session_id else int(time.time() * 1000)
    except (ValueError, TypeError):
        sid = int(time.time() * 1000)
    print(f"[DEBUG] session_id from frontend: {session_id}, using sid: {sid}") 

    results = []

    if message and message.strip():
        lines = [l.strip() for l in message.strip().splitlines() if l.strip()]
        from utils.source_detector import _looks_like_bare_url

        urls      = [l for l in lines if l.startswith("http://") or l.startswith("https://") or _looks_like_bare_url(l)]
        plain     = [l for l in lines if l not in urls]
        plain_str = "\n".join(plain).strip()

        for url in urls:
            try:
                result = await process_link(url, session_id=sid)
                if result: results.append(result)
            except Exception as e:
                results.append(f"Error processing '{url}': {e}")

        if plain_str:
            try:
                result = await process_text_input(plain_str, session_id=sid)
                if result: results.append(result)
            except Exception as e:
                results.append(f"Error processing text: {e}")

    if images:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        for img in images:
            if not img or not img.filename:
                continue
            try:
                safe_name = img.filename.strip().replace(" ", "_")
                if safe_name.lower().endswith(".pdf"):
                    save_path = PDFS_DIR / safe_name        # ← save PDFs separately
                else:
                    save_path = IMAGES_DIR / safe_name
                with open(save_path, "wb") as f:
                    shutil.copyfileobj(img.file, f)

                resolved = str(save_path.resolve())

                # ── Route PDFs to process_local_file, not process_image_input
                if safe_name.lower().endswith(".pdf"):
                    from main import process_local_file
                    result = await process_local_file(resolved)
                else:
                    result = await process_image_input(resolved, session_id=sid)

                if result: results.append(result)
            except Exception as e:
                results.append(f"Error processing '{img.filename}': {e}")


    if not results:
        return {"response": "Nothing to process. Please provide a link, text, or image."}

    return {"response": "\n\n---\n\n".join(str(r) for r in results)}

@app.patch("/api/resources/{resource_id}/answer")
async def api_update_answer(resource_id: int, payload: dict):
    row = get_resource(resource_id)
    if not row:
        raise HTTPException(status_code=404, detail="Resource not found")
    from database.db import update_resource_answer
    update_resource_answer(resource_id, payload.get("llm_output", ""))
    return {"ok": True}
