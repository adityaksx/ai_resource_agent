from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys
sys.path.append("..")

from main import process_link

app = FastAPI()

app.mount("/static", StaticFiles(directory="web/static"), name="static")


class Message(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
def home():
    with open("web/templates/chat.html") as f:
        return f.read()


@app.post("/chat")
def chat(msg: Message):

    result = process_link(msg.message)

    return {"response": result}