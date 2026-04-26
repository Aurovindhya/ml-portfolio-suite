"""
observability/langfuse_client.py

Shared Langfuse tracer used by all GenAI and agentic modules.
Degrades gracefully to a no-op tracer if keys are not configured.

Usage:
    from observability.langfuse_client import tracer

    with tracer.trace("chatbot") as span:
        span.set_input({"message": user_message})
        response = llm.invoke(user_message)
        span.set_output({"response": response})
        span.set_tokens(input=100, output=200)
"""

import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional

from core.config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

logger = logging.getLogger(__name__)


class _Span:
    def __init__(self, trace_id: str, name: str, lf_trace=None):
        self.trace_id = trace_id
        self.name = name
        self._lf = lf_trace
        self._start = time.time()

    def set_input(self, data: Dict[str, Any]):
        if self._lf:
            try: self._lf.update(input=data)
            except Exception: pass

    def set_output(self, data: Dict[str, Any]):
        if self._lf:
            try: self._lf.update(output=data)
            except Exception: pass

    def set_tokens(self, input: int = 0, output: int = 0):
        if self._lf:
            try: self._lf.update(usage={"input": input, "output": output})
            except Exception: pass

    def set_score(self, name: str, value: float, comment: str = ""):
        """Attach an eval score (e.g. relevance, groundedness) to the trace."""
        if self._lf:
            try:
                self._lf.score(name=name, value=value, comment=comment)
            except Exception: pass

    def set_metadata(self, data: Dict[str, Any]):
        if self._lf:
            try: self._lf.update(metadata=data)
            except Exception: pass

    def finish(self, error: Optional[str] = None):
        latency = (time.time() - self._start) * 1000
        if self._lf:
            try:
                update: Dict[str, Any] = {"metadata": {"latency_ms": round(latency, 2)}}
                if error:
                    update["level"] = "ERROR"
                    update["status_message"] = error
                self._lf.update(**update)
            except Exception: pass


class _NoopTracer:
    @contextmanager
    def trace(self, name: str = "call"):
        span = _Span(trace_id=str(uuid.uuid4())[:8], name=name)
        try:
            yield span
        except Exception as e:
            span.finish(error=str(e))
            raise
        else:
            span.finish()


class LangfuseTracer:
    def __init__(self, client):
        self._client = client

    @contextmanager
    def trace(self, name: str = "call"):
        trace_id = str(uuid.uuid4())
        lf_trace = self._client.trace(id=trace_id, name=name)
        span = _Span(trace_id=trace_id[:8], name=name, lf_trace=lf_trace)
        try:
            yield span
        except Exception as e:
            span.finish(error=str(e))
            raise
        else:
            span.finish()


def _build_tracer():
    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        try:
            from langfuse import Langfuse
            client = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST,
            )
            logger.info("Langfuse tracing enabled.")
            return LangfuseTracer(client)
        except Exception as e:
            logger.warning(f"Langfuse init failed, using noop tracer: {e}")
    logger.info("Langfuse keys not set — tracing disabled.")
    return _NoopTracer()


tracer = _build_tracer()
