"""AasthaGPT backend (FastAPI) - runs locally and on Vercel."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from groq import Groq
from pydantic import BaseModel, Field
from pypdf import PdfReader

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

BASE_DIR = Path(__file__).resolve().parent.parent
RESUME_PATH = BASE_DIR / "A_Resume.pdf"
INDEX_HTML = BASE_DIR / "index.html"

app = FastAPI(title="AasthaGPT", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ----------------------------- models --------------------------------

class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(max_length=1000)
    history: list[Turn] = Field(default_factory=list)


# ----------------------------- resume --------------------------------

@lru_cache(maxsize=1)
def get_resume_text() -> str:
    """Read the resume PDF once per server instance."""
    if not RESUME_PATH.exists():
        raise FileNotFoundError(f"Resume PDF not found: {RESUME_PATH}")

    reader = PdfReader(str(RESUME_PATH))
    text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()

    if not text:
        raise ValueError("Could not extract any text from the resume PDF.")
    return text


@lru_cache(maxsize=1)
def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    return Groq(api_key=api_key)


def build_system_prompt(resume_text: str) -> str:
    return f"""
You are AasthaGPT, an AI assistant that represents the candidate Aastha Garg
and answers questions from recruiters and interviewers about her.

CANDIDATE RESUME (the only source of facts about her):
\"\"\"
{resume_text}
\"\"\"

RULES:
1. Never invent qualifications, work history, achievements, certifications,
   or personal experiences. Only use facts from the resume.
2. "Tell me about yourself": write a natural introduction from her education,
   skills, projects and goals.
3. "Why should we hire you?": connect her real skills and projects to what a
   fresher employer needs.
4. Strengths: base them on documented skills and projects.
5. Weaknesses: give a constructive fresher-level sample answer and say it is
   a suggested answer, not a verified fact.
6. Challenges, failures, conflicts: never invent a story. Say details are not
   in the resume and offer a short framework (situation, action, result).
7. Technical questions: explain correctly and simply.
8. Do not reply "I don't have enough information" to ordinary HR questions.
9. If something truly is not in the resume, say so briefly.
10. Be clear, concise and professional. Refer to her as "Aastha" / "she".
11. Format: short paragraphs and simple "- " bullet lists. No tables, no headings.
"""


# ----------------------------- routes --------------------------------

api = APIRouter(prefix="/api")


@api.get("/health")
def health():
    return {"status": "healthy"}


@api.post("/chat")
def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        client = get_client()
        resume_text = get_resume_text()
    except Exception as exc:  # missing key / missing PDF
        print(f"Startup error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    messages = [{"role": "system", "content": build_system_prompt(resume_text)}]
    messages += [t.model_dump() for t in request.history[-8:]]
    messages.append({"role": "user", "content": question})

    def stream():
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.4,
                # gpt-oss "thinks" first and those tokens count toward this limit
                max_completion_tokens=2000,
                stream=True,
                extra_body={"reasoning_effort": "low"},
            )
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as exc:
            print(f"Groq error: {exc}")
            yield "\n\nSorry, something went wrong while generating the answer. Please try again."

    return StreamingResponse(
        stream(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app.include_router(api)


@app.get("/")
def home():
    """Local dev convenience. On Vercel, index.html is served statically."""
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return {"message": "AasthaGPT backend is running", "status": "ok"}
