"""
modules/generative/pdf_assistant/model.py

PDF Question-Answering using Retrieval-Augmented Generation (RAG).

Pipeline:
  1. PDF → text chunks (semantic splitting, 500 tokens, 50 overlap)
  2. Chunks → embeddings via Gemini embedding model
  3. Embeddings → FAISS vector index
  4. Query → retrieve top-K chunks → Gemini Pro generates answer
  5. Full trace logged to Langfuse

Usage:
    from modules.generative.pdf_assistant.model import PDFAssistant
    assistant = PDFAssistant()
    assistant.ingest(pdf_bytes)
    answer = assistant.ask("What are the key findings?")
"""

import io
import time
from typing import List, Dict, Optional

from core.config import GEMINI_API_KEY
from observability.langfuse_client import tracer

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


CHUNK_SIZE    = 500   # characters
CHUNK_OVERLAP = 50
TOP_K         = 4     # chunks to retrieve
EMBED_MODEL   = "models/embedding-001"
GEN_MODEL     = "gemini-pro"

RAG_PROMPT_TEMPLATE = """You are an expert document analyst. Answer the question using ONLY the context below.
If the answer is not in the context, say "I could not find this in the document."

Context:
{context}

Question: {question}

Answer:"""


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


class PDFAssistant:
    def __init__(self):
        if not GENAI_AVAILABLE:
            raise ImportError("google-generativeai not installed.")
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")
        if not PYPDF_AVAILABLE:
            raise ImportError("pypdf not installed. Run: pip install pypdf")
        if not GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY not set.")

        genai.configure(api_key=GEMINI_API_KEY)
        self._gen_model  = genai.GenerativeModel(GEN_MODEL)
        self._index: Optional[faiss.IndexFlatL2] = None
        self._chunks: List[str] = []
        self.doc_name: Optional[str] = None

    def ingest(self, pdf_bytes: bytes, doc_name: str = "document.pdf"):
        """Parse PDF, chunk text, embed, and build FAISS index."""
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)

        if not raw_text.strip():
            raise ValueError("PDF appears to be empty or image-only (no extractable text).")

        self._chunks  = _chunk_text(raw_text)
        self.doc_name = doc_name

        # Embed all chunks
        embeddings = []
        for chunk in self._chunks:
            result = genai.embed_content(model=EMBED_MODEL, content=chunk, task_type="retrieval_document")
            embeddings.append(result["embedding"])

        matrix = np.array(embeddings, dtype="float32")
        dim    = matrix.shape[1]
        self._index = faiss.IndexFlatL2(dim)
        self._index.add(matrix)

    def ask(self, question: str) -> Dict:
        if self._index is None:
            raise RuntimeError("No document ingested. Call ingest() first.")

        with tracer.trace("pdf_assistant") as span:
            span.set_input({"question": question, "doc": self.doc_name, "chunks": len(self._chunks)})

            # Embed query
            q_embed = genai.embed_content(model=EMBED_MODEL, content=question, task_type="retrieval_query")
            q_vec   = np.array([q_embed["embedding"]], dtype="float32")

            # Retrieve
            _, indices = self._index.search(q_vec, TOP_K)
            retrieved  = [self._chunks[i] for i in indices[0] if i < len(self._chunks)]
            context    = "\n\n---\n\n".join(retrieved)

            # Generate
            prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
            t0 = time.time()
            response = self._gen_model.generate_content(prompt)
            ms = (time.time() - t0) * 1000
            answer = response.text

            input_tokens  = getattr(response.usage_metadata, "prompt_token_count", 0)
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
            span.set_tokens(input=input_tokens, output=output_tokens)
            span.set_output({"answer": answer[:200]})

            return {
                "answer":        answer,
                "sources":       retrieved,
                "doc_name":      self.doc_name,
                "chunks_used":   len(retrieved),
                "input_tokens":  input_tokens,
                "output_tokens": output_tokens,
                "trace_id":      span.trace_id,
                "inference_ms":  round(ms, 2),
            }


# Per-session assistant store
_assistants: Dict[str, PDFAssistant] = {}


def get_assistant(session_id: str) -> PDFAssistant:
    if session_id not in _assistants:
        _assistants[session_id] = PDFAssistant()
    return _assistants[session_id]
