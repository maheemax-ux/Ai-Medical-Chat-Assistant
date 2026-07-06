

import os
import pickle
import numpy as np

from config import (
    EMBEDDING_MODEL_NAME,
    TOP_K_RESULTS,
    VECTOR_INDEX_PATH,
    MEDICAL_DISCLAIMER,
)
from rag.knowledge_base import build_chunks
from rag.llm_client import LLMClient
from rag.query_filter import is_medical_query
from rag.web_search import fetch_web_context


class RAGEngine:
    def __init__(self):
        self._embedder = None
        self._embeddings = None
        self._chunks = []
        self.llm = LLMClient()
        self._load_or_build_index()


    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self._embedder

    def _embed(self, texts: list) -> np.ndarray:
        vecs = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
      
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        return vecs / norms

    def _load_or_build_index(self):
        chunks = build_chunks()
        self._chunks = chunks

        if not chunks:
            self._embeddings = None
            return

        emb_file = VECTOR_INDEX_PATH + ".npy"
        meta_file = VECTOR_INDEX_PATH + ".meta.pkl"

        os.makedirs(os.path.dirname(VECTOR_INDEX_PATH), exist_ok=True)


        if os.path.exists(emb_file) and os.path.exists(meta_file):
            try:
                with open(meta_file, "rb") as f:
                    cached_chunks = pickle.load(f)
                if len(cached_chunks) == len(chunks):
                    self._embeddings = np.load(emb_file)
                    self._chunks = cached_chunks
                    return
            except Exception:
                pass 

        texts = [c["text"] for c in chunks]
        embeddings = self._embed(texts)

        np.save(emb_file, embeddings)
        with open(meta_file, "wb") as f:
            pickle.dump(chunks, f)

        self._embeddings = embeddings

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS):
        if self._embeddings is None or not self._chunks:
            return []
        q_vec = self._embed([query])[0]  # shape (dim,)
        scores = self._embeddings @ q_vec  # cosine similarity (both L2-normalized)
        top_idx = np.argsort(-scores)[:top_k]
        results = []
        for idx in top_idx:
            chunk = self._chunks[idx]
            results.append({**chunk, "score": float(scores[idx])})
        return results

    def answer(self, query: str, chat_history: list = None, use_web: bool = True) -> dict:
        """
        Returns {"answer": str, "sources": [...], "blocked": bool}
        """
        if not is_medical_query(query, llm_client=self.llm):
            return {
                "answer": (
                    "I'm a medical-assistant chatbot and can only help with "
                    "health, disease, symptom, or medication related "
                    "questions. Please ask something related to health or "
                    "medicine."
                ),
                "sources": [],
                "blocked": True,
            }

        retrieved = self.retrieve(query)

        web_results = fetch_web_context(query) if use_web else []
        combined = retrieved + [
            {"source": w["source"], "text": w["text"], "score": None} for w in web_results
        ]

        source_labels = [r["source"] for r in retrieved] + [
            f"{w['source']} ({w['url']})" if w.get("url") else w["source"] for w in web_results
        ]

        if not self.llm.is_configured:
            if not combined:
                answer_text = (
                    "No AI model is configured yet, and no matching information "
                    "was found in the local knowledge base or live medical "
                    "sources for this question. Set LLM_PROVIDER (and an API "
                    "key) in config.py to enable AI-generated answers, or add "
                    "more reference documents to knowledge_base/."
                )
            else:
                sections = []
                for i, r in enumerate(combined, 1):
                    source = r.get('source', 'Unknown')
                    text = r.get('text', '').strip()
                    # Ensure text has proper line breaks for markdown
                    # Replace single newlines with double newlines for markdown paragraph breaks
                    formatted_text = text.replace('\n', '  \n')
                    sections.append(
                        f"### 📄 Source {i}: {source}\n\n"
                        f"{formatted_text}"
                    )
                bullets = "\n\n---\n\n".join(sections)
                answer_text = (
                    "> ⚠️ *No AI model is configured — showing the raw information "
                    "found for your question instead of a generated summary. "
                    "Set `LLM_PROVIDER` and an API key in `config.py` to enable "
                    "AI-written answers.*\n\n"
                    + bullets
                )
            return {
                "answer": f"{answer_text}\n\n{MEDICAL_DISCLAIMER}",
                "sources": source_labels,
                "blocked": False,
            }

        context_text = "\n\n".join(
            f"[Source: {r['source']}]\n{r['text']}" for r in combined
        )

        system_prompt = (
            "You are a careful, evidence-based medical information assistant. "
            "Answer the user's health question using the provided context "
            "(which may include local reference notes, PubMed abstracts, and "
            "MedlinePlus summaries) when it's relevant. If the context doesn't "
            "fully cover the question, say so and give general, cautious "
            "medical information. Never provide a definitive diagnosis — "
            "always recommend consulting a licensed healthcare professional "
            "for diagnosis or treatment decisions.\n\n"
            "IMPORTANT FORMATTING RULES:\n"
            "- Provide a DETAILED and THOROUGH answer using multiple paragraphs.\n"
            "- Use markdown formatting: headings (##, ###), bullet points, bold text.\n"
            "- Structure your answer with clear sections like: Overview, Causes, "
            "Symptoms, Treatment, When to See a Doctor.\n"
            "- Include all relevant information from the provided context.\n"
            "- Mention when information comes from recent medical literature vs general knowledge.\n"
            "- Do NOT give a single-line or single-paragraph answer. Always provide "
            "a comprehensive, well-organized response."
        )

        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            messages.extend(chat_history[-6:])  # last few turns for continuity
        messages.append({"role": "user", "content": user_prompt})

        reply = self.llm.chat(messages)

        # If the LLM call failed, fall back to showing retrieved context
        if reply.startswith("[LLM ERROR]") or reply.startswith("[NO_LLM]"):
            error_msg = reply
            if combined:
                sections = []
                for i, r in enumerate(combined, 1):
                    source = r.get('source', 'Unknown')
                    text = r.get('text', '').strip()
                    formatted_text = text.replace('\n', '  \n')
                    sections.append(
                        f"### 📄 Source {i}: {source}\n\n"
                        f"{formatted_text}"
                    )
                context_display = "\n\n---\n\n".join(sections)
                answer_text = (
                    f"> ⚠️ **LLM Error:** {error_msg}\n>\n"
                    f"> *Showing retrieved information instead:*\n\n"
                    + context_display
                )
            else:
                answer_text = (
                    f"⚠️ **LLM Error:** {error_msg}\n\n"
                    "No matching information was found in the knowledge base either."
                )
            return {
                "answer": f"{answer_text}\n\n{MEDICAL_DISCLAIMER}",
                "sources": source_labels,
                "blocked": False,
            }

        reply_with_disclaimer = f"{reply}\n\n{MEDICAL_DISCLAIMER}"

        return {
            "answer": reply_with_disclaimer,
            "sources": source_labels,
            "blocked": False,
        }
