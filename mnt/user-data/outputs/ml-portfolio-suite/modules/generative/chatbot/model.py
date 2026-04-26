"""
modules/generative/chatbot/model.py

Multi-turn conversational agent powered by Gemini Pro.
Features:
  - Persistent conversation memory (in-memory per session)
  - Configurable system prompt
  - Langfuse tracing (input tokens, output tokens, latency)
  - Structured response with metadata

Usage:
    from modules.generative.chatbot.model import ChatSession
    session = ChatSession()
    response = session.chat("What is transfer learning?")
"""

import time
import uuid
from typing import List, Dict, Optional

from core.config import GEMINI_API_KEY
from observability.langfuse_client import tracer

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable AI assistant specialising in machine learning, "
    "data science, and software engineering. Be concise, accurate, and helpful."
)


class ChatSession:
    """
    Stateful chat session with Gemini Pro.
    Each instance maintains its own conversation history.
    """

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        if not GENAI_AVAILABLE:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        if not GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY not set in .env")

        genai.configure(api_key=GEMINI_API_KEY)
        self.session_id  = str(uuid.uuid4())[:8]
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []
        self._model = genai.GenerativeModel(
            model_name="gemini-pro",
            system_instruction=system_prompt,
        )
        self._chat = self._model.start_chat(history=[])

    def chat(self, message: str) -> Dict:
        """
        Send a message and return a structured response dict.
        Traces the call to Langfuse.
        """
        with tracer.trace("chatbot") as span:
            span.set_input({"message": message, "session_id": self.session_id})

            t0 = time.time()
            response = self._chat.send_message(message)
            ms = (time.time() - t0) * 1000

            reply = response.text
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "model", "content": reply})

            # Token counts (available if Gemini returns usage metadata)
            input_tokens  = getattr(response.usage_metadata, "prompt_token_count",  0)
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
            span.set_tokens(input=input_tokens, output=output_tokens)
            span.set_output({"reply": reply[:200]})

            return {
                "reply":          reply,
                "session_id":     self.session_id,
                "turn":           len(self.history) // 2,
                "input_tokens":   input_tokens,
                "output_tokens":  output_tokens,
                "trace_id":       span.trace_id,
                "inference_ms":   round(ms, 2),
            }

    def reset(self):
        """Clear conversation history."""
        self.history = []
        self._chat = self._model.start_chat(history=[])


# Module-level session store (per-session_id)
_sessions: Dict[str, ChatSession] = {}


def get_or_create_session(session_id: Optional[str] = None, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> ChatSession:
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    session = ChatSession(system_prompt=system_prompt)
    _sessions[session.session_id] = session
    return session
