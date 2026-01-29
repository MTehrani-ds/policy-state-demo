from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field


# ---------------------------
# State machine models
# ---------------------------

class Step(str, Enum):
    LIMITED_RESPONSE = "LIMITED_RESPONSE"
    COLLECT_CONTACT = "COLLECT_CONTACT"
    HANDOFF = "HANDOFF"


class Collected(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    best_time: Optional[str] = None


class SessionState(BaseModel):
    step: Step = Step.LIMITED_RESPONSE
    procedure: Optional[str] = None
    intent: Optional[str] = None
    collected: Collected = Field(default_factory=Collected)


class IncomingMessage(BaseModel):
    session_id: str
    user_message: str
    channel: str = "webchat"
    practice_name: str = "Example Dental Clinic"
    prior_state: Optional[SessionState] = None
    msg: Optional[str] = None
    state: Optional[SessionState] = None


class OutgoingMessage(BaseModel):
    session_id: str
    channel: str
    practice_name: str
    user_message: str
    reply: str
    state: SessionState


# ---------------------------
# App + error handler
# ---------------------------

app = FastAPI(title="Policy+StateMachine Demo Backend", version="0.2.0")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logging.exception("Unhandled server error")
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(exc)})


# ---------------------------
# In-memory session store
# ---------------------------

SESSION_DB: Dict[str, SessionState] = {}


def get_state(payload: IncomingMessage) -> SessionState:
    if payload.state is not None:
        return payload.state
    if payload.prior_state is not None:
        return payload.prior_state
    if payload.session_id in SESSION_DB:
        return SESSION_DB[payload.session_id]
    return SessionState()


def save_state(session_id: str, state: SessionState) -> None:
    SESSION_DB[session_id] = state


# ---------------------------
# Policy layer
# ---------------------------

def is_medical_or_medication_question(text: str) -> bool:
    t = text.lower()
    keywords = [
        "antibiotic", "antibiotics", "amoxicillin", "penicillin", "clindamycin",
        "medicine", "medication", "dose", "dosage", "should i",
        "ibuprofen", "painkiller", "prescription",
    ]
    return any(k in t for k in keywords)


def limited_response_policy(practice_name: str) -> str:
    return (
        f"Thanks for your question. I can’t recommend specific medication (including antibiotics) "
        f"without a clinician evaluating your situation.\n\n"
        f"If you have severe swelling, fever, trouble swallowing/breathing, or rapidly worsening pain, "
        f"please seek urgent care immediately.\n\n"
        f"If not urgent: the safest next step is to speak with a dentist from {practice_name}. "
        f"Can I take your **name** and **phone number**, and the **best time** to call you back?"
    )


# ---------------------------
# Extraction (hardened)
# ---------------------------

def update_collected_from_text(state: SessionState, user_text: str) -> SessionState:
    if state.collected is None:
        state.collected = Collected()

    text = user_text.strip()
    t = text.lower()

    # Phone: naive parse
    digits = "".join(ch for ch in text if ch.isdigit() or ch == "+")
    if len(digits.replace("+", "")) >= 9 and not state.collected.phone:
        state.collected.phone = digits

    # Best time: naive phrases
    if not state.collected.best_time and any(x in t for x in ["morning", "afternoon", "evening", "today", "tomorrow", "anytime"]):
        state.collected.best_time = text

    # Name: case-safe, no split indexing
    if not state.collected.name:
        patterns = ["my name is ", "i am ", "i'm "]
        for pat in patterns:
            idx = t.find(pat)
            if idx != -1:
                name = text[idx + len(pat):].strip()
                for stop in [".", ",", ";", " and "]:
                    if stop in name:
                        name = name.split(stop, 1)[0].strip()
                if len(name) >= 2:
                    state.collected.name = name[:80]
                break

    return state


# ---------------------------
# State machine
# ---------------------------

def next_reply(practice_name: str, user_text: str, state: SessionState) -> tuple[str, SessionState]:
    if is_medical_or_medication_question(user_text):
        state.step = Step.LIMITED_RESPONSE
        return limited_response_policy(practice_name), state

    if state.step in (Step.LIMITED_RESPONSE, Step.COLLECT_CONTACT):
        state = update_collected_from_text(state, user_text)

        missing = []
        if not state.collected.name:
            missing.append("name")
        if not state.collected.phone:
            missing.append("phone number")
        if not state.collected.best_time:
            missing.append("best time to call")

        if missing:
            state.step = Step.COLLECT_CONTACT
            return (
                "To arrange a callback, I still need your " + ", ".join(missing) +
                ". You can reply in one message like: “My name is …, phone …, best time …”.",
                state,
            )

        state.step = Step.HANDOFF
        return (
            f"Thanks, {state.collected.name}. I’ve captured your details.\n\n"
            f"Phone: {state.collected.phone}\n"
            f"Best time: {state.collected.best_time}\n\n"
            f"Someone from {practice_name} will contact you.",
            state,
        )

    return (
        f"Thanks — your request is with the team at {practice_name}.",
        state,
    )


# ---------------------------
# Routes
# ---------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/webchat/message", response_model=OutgoingMessage)
def webchat_message(payload: IncomingMessage) -> OutgoingMessage:
    state = get_state(payload)
    reply, new_state = next_reply(payload.practice_name, payload.user_message, state)
    save_state(payload.session_id, new_state)

    return OutgoingMessage(
        session_id=payload.session_id,
        channel=payload.channel,
        practice_name=payload.practice_name,
        user_message=payload.user_message,
        reply=reply,
        state=new_state,
    )


@app.post("/admin/reset_session/{session_id}")
def reset_session(session_id: str) -> Dict[str, Any]:
    SESSION_DB.pop(session_id, None)
    return {"ok": True, "session_id": session_id}


# Chat UI (serves chat.html if present)
@app.get("/", response_class=HTMLResponse)
def home():
    try:
        with open("chat.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h3>chat.html not found</h3><p>Create chat.html next to main.py.</p>"
