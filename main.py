from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
import os
import re
import uvicorn

from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in your .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Agri Conversational Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schema ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str
    message: str

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty.")
        if len(v) > 2000:
            raise ValueError("Message exceeds 2000 characters.")
        return v

    @field_validator("user_id")
    @classmethod
    def user_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("user_id cannot be empty.")
        return v.strip()


# ── Memory ────────────────────────────────────────────────────────────────────

conversation_memory: dict[str, list[dict]] = {}

MAX_HISTORY_TURNS = 6          # keep last 6 messages (3 turns)
MAX_USERS_CACHED  = 1_000      # evict oldest when cache grows too large


def get_memory(user_id: str) -> list[dict]:
    return conversation_memory.get(user_id, [])


def update_memory(user_id: str, messages: list[dict]) -> None:
    if len(conversation_memory) >= MAX_USERS_CACHED and user_id not in conversation_memory:
        # evict the oldest entry
        oldest = next(iter(conversation_memory))
        del conversation_memory[oldest]
    conversation_memory[user_id] = messages[-MAX_HISTORY_TURNS:]


# ── Off-topic guard ───────────────────────────────────────────────────────────

# Keywords that strongly indicate off-topic requests
_OFF_TOPIC_PATTERNS = re.compile(
    r"\b("
    r"politic|election|vote|president|government|war|military|weapon|"
    r"crypto|bitcoin|stock|invest|finance|bank|loan|"
    r"movie|music|sport|football|soccer|basketball|game|"
    r"celebrity|gossip|fashion|beauty|makeup|"
    r"code|programming|software|hack|"
    r"religion|god|pray|church|mosque|"
    r"relationship|dating|love|sex|"
    r"medical|doctor|hospital|drug|medicine|diagnos"
    r")\b",
    re.IGNORECASE,
)

# Lightweight keyword list for farming relevance
_FARMING_KEYWORDS = re.compile(
    r"\b("
    r"farm|crop|soil|plant|seed|fertiliz|irrigat|pest|weed|harvest|"
    r"livestock|cattle|poultry|chicken|goat|pig|fish|aqua|"
    r"rain|drought|climate|weather|season|compost|organic|yield|"
    r"market|storage|silo|greenhouse|tractor|land|field|garden|"
    r"cassava|maize|rice|yam|tomato|vegetable|fruit|coco|palm|"
    r"disease|blight|fungus|insect|spray|agro|agriculture|agric"
    r")\b",
    re.IGNORECASE,
)


def is_off_topic(message: str) -> bool:
    """
    Returns True if the message is clearly not related to farming.
    Strategy: flag if it matches off-topic patterns AND contains no farming keywords.
    """
    has_off_topic  = bool(_OFF_TOPIC_PATTERNS.search(message))
    has_farming    = bool(_FARMING_KEYWORDS.search(message))
    return has_off_topic and not has_farming


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are AgriBot, a dedicated agricultural assistant for small and medium-scale farmers,
primarily in developing regions (Africa, South Asia, Southeast Asia, Latin America).

╔══════════════════════════════════════════════════════════════╗
║  YOUR SCOPE — STRICTLY ENFORCED                              ║
╠══════════════════════════════════════════════════════════════╣
║  You ONLY discuss topics that are directly related to        ║
║  farming and agriculture, including:                         ║
║  • Crop selection, planting calendars, spacing, rotation     ║
║  • Soil health, pH, composting, organic matter               ║
║  • Pest and disease identification and management            ║
║  • Irrigation techniques and water management                ║
║  • Fertilizer use (organic-first, then synthetic)            ║
║  • Post-harvest handling, storage, and value addition        ║
║  • Livestock and poultry basics                              ║
║  • Climate-smart and sustainable farming practices           ║
║  • Agribusiness: pricing, cooperative farming, market access ║
╠══════════════════════════════════════════════════════════════╣
║  YOU MUST REFUSE ALL OFF-TOPIC REQUESTS.                     ║
║  If a message is not about farming/agriculture, reply ONLY:  ║
║  "I'm your farming assistant and can only help with          ║
║   agricultural questions. Please ask me something related    ║
║   to crops, soil, pests, livestock, or farming practices."  ║
║  Do NOT apologise repeatedly. Do NOT engage with the         ║
║  off-topic subject at all.                                   ║
╠══════════════════════════════════════════════════════════════╣
║  RESPONSE STYLE                                              ║
║  • Practical, affordable, and locally applicable advice      ║
║  • Plain language — assume limited formal education          ║
║  • Recommend organic/natural solutions first                 ║
║  • Never recommend unsafe chemical dosages                   ║
║  • Keep answers concise; use numbered steps when needed      ║
╚══════════════════════════════════════════════════════════════╝

Begin every response by checking: is this question about farming?
If YES → answer helpfully.
If NO  → use the refusal message above, nothing else.
""".strip()


# ── Route ─────────────────────────────────────────────────────────────────────

@app.post("/chat")
def chat(request: ChatRequest):
    # Fast local off-topic check before hitting the API
    if is_off_topic(request.message):
        return {
            "reply": (
                "I'm your farming assistant and can only help with agricultural questions. "
                "Please ask me something related to crops, soil, pests, livestock, or farming practices."
            )
        }

    history = get_memory(request.user_id)

    # Build the conversation text
    conversation_lines = [SYSTEM_PROMPT, ""]
    for msg in history:
        role_label = "Farmer" if msg["role"] == "user" else "AgriBot"
        conversation_lines.append(f"{role_label}: {msg['content']}")
    conversation_lines.append(f"Farmer: {request.message}")
    conversation_lines.append("AgriBot:")

    full_prompt = "\n".join(conversation_lines)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
        )
        ai_reply = response.text.strip()
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"AI backend error: {str(e)}"
        )

    # Persist memory
    history.append({"role": "user",      "content": request.message})
    history.append({"role": "assistant", "content": ai_reply})
    update_memory(request.user_id, history)

    return {"reply": ai_reply}


@app.delete("/chat/{user_id}/history")
def clear_history(user_id: str):
    """Clear conversation history for a specific user."""
    conversation_memory.pop(user_id, None)
    return {"status": "history cleared", "user_id": user_id}


@app.get("/")
def health_check():
    return {"status": "AgriBot is up and running 🌱"}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,          # auto-reload on file changes (dev mode)
        log_level="info",
    )