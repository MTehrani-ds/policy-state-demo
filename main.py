"""
single_file_demo_backend.py

FastAPI single-file demo backend that:
- translates Policy + State Machine logic into Python
- keeps a stable "Postman-friendly" JSON shape (session object in, session object out)
- deploys fast in ~15 minutes (uvicorn)

RUN:
  pip install fastapi "uvicorn[standard]" pydantic
  uvicorn single_file_demo_backend:app --host 0.0.0.0 --port 8000 --reload

POSTMAN:
  POST http://localhost:8000/webchat/message
  Body (JSON): see example at bottom.
"""

from __future__ import annotations
from fastapi.responses import HTMLResponse
from enum import Enum
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field


# ---------------------------
# Policy + State Machine
# ---------------------------

class Step(str, Enum):
    LIMITED_RESPONSE = "LIMITED_RESPONSE"          # safe, no medical instructions
    COLLECT_CONTACT = "COLLECT_CONTACT"            # ask for name/phone/best time
    HANDOFF = "HANDOFF"                            # we have contact -> next is staff follow-up


class Collected(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    best_time: Optional[str] = None


class SessionState(BaseModel):
    step: Step = Step.LIMITED_RESPONSE
    procedure: Optional[str] = None
    intent: Optional[str] = None
    collected: Collected = Field(default_factory=Collected)


# This is the "shape" I’m keeping stable:
# - session_id, channel, practice_name, prior_state, user_message, msg, state
#   (matches your earlier example closely)
class IncomingMessage(BaseModel):
    session_id: str
    user_message: str
    channel: str = "webchat"
    practice_name: str = "Example Dental Clinic"
    prior_state: Optional[SessionState] = None
    msg: Optional[str] = None
    state: Optional[SessionState] = None  # allow Postman to send state in this field too


class OutgoingMessage(BaseModel):
    session_id: str
    channel: str
    practice_name: str
    user_message: str
    reply: str
    state: SessionState


# ---------------------------
# In-memory session store
# ---------------------------

SESSION_DB: Dict[str, SessionState] = {}


def get_state(payload: IncomingMessage) -> SessionState:
    # Priority: payload.state -> payload.prior_state -> stored -> default
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
# Policy layer (safe responses)
# ---------------------------

def is_medical_or_medication_question(text: str) -> bool:
    t = text.lower()
    keywords = [
        "antibiotic", "antibiotics", "amoxicillin", "penicillin", "clindamycin",
        "medicine", "medication", "dose", "dosage", "take", "should i",
        "ibuprofen", "painkiller", "prescription",
    ]
    return any(k in t for k in keywords)


def limited_response_policy(practice_name: str, user_text: str) -> str:
    # Conservative, industry-safe: no diagnosing, no prescribing, no dosage.
    # Offer: general guidance + escalation + capture contact.
    return (
        f"Thanks for your question. I can’t recommend specific medication (including antibiotics) "
        f"without a clinician evaluating your situation.\n\n"
        f"If you have severe swelling, fever, trouble swallowing/breathing, or rapidly worsening pain, "
        f"please seek urgent care immediately.\n\n"
        f"If not urgent: the safest next step is to speak with a dentist from {practice_name} for proper guidance. "
        f"Can I take your **name** and **phone number**, and the **best time** to call you back?"
    )


# ---------------------------
# State machine transitions
# ---------------------------

def update_collected_from_text(state: SessionState, user_text: str) -> SessionState:
    """
    Lightweight demo extraction (no AI):
    - phone: detect sequences with + and digits
    - name / best_time: naive heuristics
    For production, you’d swap this with an NER or structured form collection.
    """
    text = user_text.strip()

    # Phone: naive parse
    digits = "".join(ch for ch in text if ch.isdigit() or ch == "+")
    if len(digits.replace("+", "")) >= 9 and state.collected.phone is None:
        state.collected.phone = digits

    # Best time: common phrases
    t = text.lower()
    if state.collected.best_time is None and any(x in t for x in ["morning", "afternoon", "evening", "today", "tomorrow", "anytime"]):
        state.collected.best_time = text

    # Name: super naive (if user writes "I am X" / "I'm X" / "My name is X")
    if state.collected.name is None:
        for pat in ["my name is ", "i am ", "i'm "]:
            if pat in t:
                name = text.split(pat, 1)[1].strip()
                if len(name) >= 2:
                    # stop at first punctuation
                    for stop in [".", ",", ";", " and "]:
                        if stop in name:
                            name = name.split(stop, 1)[0].strip()
                    state.collected.name = name[:80]
                break

    return state


def next_reply(practice_name: str, user_text: str, state: SessionState) -> tuple[str, SessionState]:
    """
    Core: policy gating + step-based dialogue.
    """
    # 1) Policy gate: if medical/medication question comes in at any time -> LIMITED_RESPONSE track
    if is_medical_or_medication_question(user_text):
        state.step = Step.LIMITED_RESPONSE
        return limited_response_policy(practice_name, user_text), state

    # 2) State machine
    if state.step == Step.LIMITED_RESPONSE:
        # We asked for name/phone/best time. Try to extract; if still missing, keep collecting.
        state = update_collected_from_text(state, user_text)

        missing = []
        if not state.collected.name:
            missing.end("name")
        if not state.collected.phone:
            missing.end("phone number")
        if not state.collected.best_time:
            missing.end("best time to call")

        if missing:
            ask = ", ".join(missing)
            reply = (
                f"Got it — to arrange a callback, I still need your {ask}. "
                f"You can reply in one message like: “My name is …, my phone is …, best time is …”."
            )
            state.step = Step.COLLECT_CONTACT
            return reply, state

        # All collected -> handoff
        state.step = Step.HANDOFF
        reply = (
            f"Thanks, {state.collected.name}. I’ve captured your details.\n\n"
            f"**Phone:** {state.collected.phone}\n"
            f"**Best time:** {state.collected.best_time}\n\n"
            f"Someone from {practice_name} will contact you. If your symptoms worsen (swelling, fever, "
            f"trouble swallowing/breathing), please seek urgent care."
        )
        return reply, state

    if state.step == Step.COLLECT_CONTACT:
        state = update_collected_from_text(state, user_text)

        missing = []
        if not state.collected.name:
            missing.end("name")
        if not state.collected.phone:
            missing.end("phone number")
        if not state.collected.best_time:
            missing.end("best time to call")

        if missing:
            return (
                "Thanks — I’m still missing: " + ", ".join(missing) + ".",
                state,
            )

        state.step = Step.HANDOFF
        return (
            f"Perfect — thanks, {state.collected.name}. We’ll call {state.collected.phone} "
            f"({state.collected.best_time}).",
            state,
        )

    # HANDOFF or unknown: keep it short
    return (
        f"Thanks — your request is with the team at {practice_name}. "
        f"If anything changes urgently, please seek immediate care.",
        state,
    )


# ---------------------------
# FastAPI 
# ---------------------------

app = FastAPI(title="Policy+StateMachine Demo Backend", version="0.1.0")

@app.get("/", response_class=HTMLResponse)
def chat_ui():
    with open("chat.html", "r", encoding="utf-8") as f:
        return f.read()


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


"""
Example Postman body:

{
  "session_id": "demo3",
  "user_message": "Should I take antibiotics?",
  "channel": "webchat",
  "practice_name": "Example Dental Clinic",
  "prior_state": null,
  "msg": "Should I take antibiotics?",
  "state": {
    "step": "LIMITED_RESPONSE",
    "procedure": null,
    "intent": null,
    "collected": { "name": null, "phone": null, "best_time": null }
  }
}

Response will echo the same envelope + updated "state".
"""
