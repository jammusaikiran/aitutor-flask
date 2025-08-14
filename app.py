


from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
import uuid
import math
import numpy as np
import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# Optional generators (pick via env RAG_PROVIDER)
RAG_PROVIDER = os.getenv("RAG_PROVIDER", "openai").lower()
USE_OPENAI = RAG_PROVIDER == "openai"
USE_GEMINI = RAG_PROVIDER == "gemini"



app = Flask(__name__)
CORS(app)

# Per-user in-memory stores
user_indices = {}  # user_id -> {'index': faiss_index, 'embeddings': np.array, 'meta': list[dict], 'dim': int}
# model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")


# Config
UPLOAD_FOLDER = "uploaded_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
DEFAULT_TOP_K = int(os.getenv("TOP_K", 5))


def normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


def extract_pdf_text_with_pages(filepath: str):
    """
    Returns list of dicts: [{'page': int, 'text': str}]
    """
    out = []
    doc = fitz.open(filepath)
    try:
        for i, page in enumerate(doc):
            txt = page.get_text("text")
            txt = re.sub(r'\s+', ' ', txt).strip()
            if txt:
                out.append({"page": i + 1, "text": txt})
    finally:
        doc.close()
    return out


def split_into_chunks(text: str, page: int, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
    """
    Overlapping character-based chunking with light sentence awareness.
    """
    # Gentle sentence split to avoid breaking mid-sentence
    sentences = re.split(r'(?<=[.?!])\s+', text)
    pieces = []
    buf = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(buf) + len(s) + 1 <= chunk_size:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                pieces.append(buf)
            buf = s
    if buf:
        pieces.append(buf)

    # Add overlap by merging adjacent pieces into sliding windows
    chunks = []
    if not pieces:
        return chunks
    i = 0
    while i < len(pieces):
        cur = pieces[i]
        j = i + 1
        while j < len(pieces) and len(cur) + 1 + len(pieces[j]) <= chunk_size:
            cur = cur + " " + pieces[j]
            j += 1
        chunks.append(cur.strip())
        # Back off by overlap proportion (approx by chars)
        if overlap > 0:
            back_chars = overlap
            # Find how many sentences to step back by roughly overlap characters
            step_text = ""
            step = j
            while step > i and len(step_text) < back_chars:
                step -= 1
                step_text = pieces[step] + " " + step_text
            i = max(i + 1, step)
        else:
            i = j

    # Add metadata
    return [{"page": page, "chunk": c} for c in chunks]


def dedupe_chunks(chunks):
    """
    Dedupe by normalized whitespace and case-insensitive hash.
    """
    seen = set()
    out = []
    for item in chunks:
        key = re.sub(r'\s+', ' ', item["chunk"]).strip().lower()
        if key not in seen and len(key) > 20:
            seen.add(key)
            out.append(item)
    return out


def build_faiss_index(embeddings: np.ndarray):
    """
    Cosine similarity via inner product on normalized vectors.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    return index


def maximal_marginal_relevance(query_vec, doc_vecs, top_k=5, lambda_mult=0.5):
    """
    Basic MMR to diversify top-k results.
    query_vec: (d,)
    doc_vecs: (n, d) normalized
    returns indices of selected docs
    """
    n = doc_vecs.shape[0]
    if n == 0:
        return []
    sim_to_query = (doc_vecs @ query_vec.reshape(-1, 1)).ravel()  # (n,)
    selected = []
    candidates = set(range(n))

    for _ in range(min(top_k, n)):
        if not selected:
            idx = int(np.argmax(sim_to_query))
            selected.append(idx)
            candidates.remove(idx)
            continue

        mmr_scores = []
        for c in candidates:
            sim_to_selected = max((doc_vecs[c] @ doc_vecs[s] for s in selected), default=0.0)
            mmr = lambda_mult * sim_to_query[c] - (1 - lambda_mult) * sim_to_selected
            mmr_scores.append((mmr, c))
        _, best_c = max(mmr_scores, key=lambda x: x[0])
        selected.append(best_c)
        candidates.remove(best_c)
    return selected


def make_prompt(question: str, contexts: list[dict]):
    """
    Build a grounded prompt with citations.
    """
    context_blocks = []
    for i, c in enumerate(contexts, 1):
        context_blocks.append(f"[{i}] (page {c['page']}) {c['chunk']}")
    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant. Answer the question strictly using the CONTEXT.
If the answer cannot be derived from the context, say "I couldn't find that in the provided documents."

CONTEXT:
{context_text}

Question: {question}

Instructions:
- Be concise and correct.
- Use only the information in CONTEXT.
- At the end, include a "Sources:" line listing the bracket numbers you used, e.g., Sources: [1,3,4].
Answer:"""
    return prompt


def generate_answer(question: str, contexts: list[dict]) -> str:
    prompt = make_prompt(question, contexts)

    merged = " ".join(c["chunk"] for c in contexts)
    if not merged:
        return "I couldn't find that in the provided documents."
    return merged[:1500] + ("\n\nSources: " + ", ".join(f"[{i+1}]" for i in range(len(contexts))))


@app.route("/train", methods=["POST"])
def train():
    try:
        file = request.files.get("file")
        user_id = request.form.get("user_id")

        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400
        if not file:
            return jsonify({"error": "Missing file"}), 400

        # Save + extract
        tmp_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.pdf")
        file.save(tmp_path)

        pages = extract_pdf_text_with_pages(tmp_path)
        os.remove(tmp_path)

        # Build chunks
        all_chunks = []
        for p in pages:
            ch = split_into_chunks(p["text"], p["page"], DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
            all_chunks.extend(ch)

        all_chunks = dedupe_chunks(all_chunks)
        if not all_chunks:
            return jsonify({"error": "No readable text extracted from PDF."}), 400

        # Embed + normalize
        texts = [c["chunk"] for c in all_chunks]
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)
        embeddings = normalize(np.asarray(embeddings, dtype=np.float32))

        # FAISS
        index = build_faiss_index(embeddings)
        index.add(embeddings)

        user_indices[user_id] = {
            "index": index,
            "embeddings": embeddings,
            "meta": all_chunks,   # list of {"page": int, "chunk": str}
            "dim": embeddings.shape[1],
        }

        return jsonify({
            "message": "Training complete",
            "chunks": len(all_chunks),
            "dimension": embeddings.shape[1],
        })
    except Exception as e:
        print("❌ Error in /train:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.get_json(force=True)
        question = data.get("question", "").strip()
        user_id = data.get("user_id")
        top_k = int(data.get("top_k", DEFAULT_TOP_K))

        if not question:
            return jsonify({"error": "Missing question"}), 400
        if not user_id or user_id not in user_indices:
            return jsonify({"error": "User index not found. Upload a PDF first."}), 400

        store = user_indices[user_id]
        index = store["index"]
        meta = store["meta"]

        # Encode + normalize query
        q_emb = model.encode([question]).astype("float32")
        q_emb = normalize(q_emb)[0]

        # Initial over-retrieve (e.g., 3x top_k) then MMR
        over_k = max(top_k * 3, top_k)
        D, I = index.search(q_emb.reshape(1, -1), over_k)
        cand_indices = I[0].tolist()

        # MMR selection on candidate set
        doc_vecs = store["embeddings"][cand_indices]
        mmr_idx_local = maximal_marginal_relevance(q_emb, doc_vecs, top_k=top_k, lambda_mult=0.6)
        selected_global = [cand_indices[i] for i in mmr_idx_local]

        contexts = [meta[i] for i in selected_global]

        answer = generate_answer(question, contexts)

        # Build citations payload
        sources = []
        for rank, gi in enumerate(selected_global, 1):
            sources.append({
                "rank": rank,
                "page": meta[gi]["page"],
                "chunk_preview": (meta[gi]["chunk"][:220] + "...") if len(meta[gi]["chunk"]) > 220 else meta[gi]["chunk"]
            })

        return jsonify({
            "answer": answer,
            "sources": sources
        })
    except Exception as e:
        print("❌ Error in /query:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/home")
def health():
    print("hello")
    return jsonify({"status": "ok", "rag_provider": RAG_PROVIDER})


if __name__ == "__main__":
    # Use a fixed port if you like: 5001
    app.run(port=5001, debug=True)
