# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import re
# import uuid
# import numpy as np
# import faiss
# import fitz  # PyMuPDF
# from dotenv import load_dotenv
# import google.generativeai as genai

# # Load environment variables
# load_dotenv()

# # Config from ENV
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     raise ValueError("Missing GEMINI_API_KEY in .env")

# # Configure Gemini
# genai.configure(api_key=GEMINI_API_KEY)
# GEMINI_EMBED_MODEL = "models/text-embedding-004"

# app = Flask(__name__)
# CORS(app)

# # Per-user in-memory stores
# user_indices = {}

# UPLOAD_FOLDER = "uploaded_docs"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))
# DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
# DEFAULT_TOP_K = int(os.getenv("TOP_K", 5))


# # ---------------- Utility Functions ---------------- #
# def normalize(vecs: np.ndarray) -> np.ndarray:
#     norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
#     return vecs / norms

# def extract_pdf_text_with_pages(filepath: str):
#     out = []
#     doc = fitz.open(filepath)
#     try:
#         for i, page in enumerate(doc):
#             txt = page.get_text("text")
#             txt = re.sub(r'\s+', ' ', txt).strip()
#             if txt:
#                 out.append({"page": i + 1, "text": txt})
#     finally:
#         doc.close()
#     return out

# def split_into_chunks(text: str, page: int, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
#     sentences = re.split(r'(?<=[.?!])\s+', text)
#     pieces, buf = [], ""
#     for s in sentences:
#         s = s.strip()
#         if not s:
#             continue
#         if len(buf) + len(s) + 1 <= chunk_size:
#             buf = (buf + " " + s).strip()
#         else:
#             if buf:
#                 pieces.append(buf)
#             buf = s
#     if buf:
#         pieces.append(buf)

#     chunks = []
#     if not pieces:
#         return chunks
#     i = 0
#     while i < len(pieces):
#         cur = pieces[i]
#         j = i + 1
#         while j < len(pieces) and len(cur) + 1 + len(pieces[j]) <= chunk_size:
#             cur = cur + " " + pieces[j]
#             j += 1
#         chunks.append(cur.strip())
#         if overlap > 0:
#             back_chars = overlap
#             step_text, step = "", j
#             while step > i and len(step_text) < back_chars:
#                 step -= 1
#                 step_text = pieces[step] + " " + step_text
#             i = max(i + 1, step)
#         else:
#             i = j
#     return [{"page": page, "chunk": c} for c in chunks]

# def dedupe_chunks(chunks):
#     seen, out = set(), []
#     for item in chunks:
#         key = re.sub(r'\s+', ' ', item["chunk"]).strip().lower()
#         if key not in seen and len(key) > 20:
#             seen.add(key)
#             out.append(item)
#     return out

# def build_faiss_index(embeddings: np.ndarray):
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dim)
#     return index

# def maximal_marginal_relevance(query_vec, doc_vecs, top_k=5, lambda_mult=0.5):
#     n = doc_vecs.shape[0]
#     if n == 0:
#         return []
#     sim_to_query = (doc_vecs @ query_vec.reshape(-1, 1)).ravel()
#     selected, candidates = [], set(range(n))
#     for _ in range(min(top_k, n)):
#         if not selected:
#             idx = int(np.argmax(sim_to_query))
#             selected.append(idx)
#             candidates.remove(idx)
#             continue
#         mmr_scores = []
#         for c in candidates:
#             sim_to_selected = max((doc_vecs[c] @ doc_vecs[s] for s in selected), default=0.0)
#             mmr = lambda_mult * sim_to_query[c] - (1 - lambda_mult) * sim_to_selected
#             mmr_scores.append((mmr, c))
#         _, best_c = max(mmr_scores, key=lambda x: x[0])
#         selected.append(best_c)
#         candidates.remove(best_c)
#     return selected

# def generate_answer(question: str, contexts: list[dict]) -> str:
#     merged = " ".join(c["chunk"] for c in contexts)
#     if not merged:
#         return "I couldn't find that in the provided documents."
#     return merged[:1500] + ("\n\nSources: " + ", ".join(f"[{i+1}]" for i in range(len(contexts))))

# def get_embeddings(texts: list[str]) -> np.ndarray:
#     """
#     Call Gemini embeddings API for a list of strings.
#     """
#     vectors = []
#     for text in texts:
#         resp = genai.embed_content(model=GEMINI_EMBED_MODEL, content=text)
#         vectors.append(resp["embedding"])
#     return normalize(np.array(vectors, dtype=np.float32))


# # ---------------- Routes ---------------- #
# @app.route("/train", methods=["POST"])
# def train():
#     try:
#         #print("File recived ", request.files.get("file"))
#         file = request.files.get("file")
#         user_id = request.form.get("user_id")
#         if not user_id:
#             return jsonify({"error": "Missing user_id"}), 400
#         if not file:
#             return jsonify({"error": "Missing file"}), 400

#         tmp_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.pdf")
#         file.save(tmp_path)
#         pages = extract_pdf_text_with_pages(tmp_path)
#         os.remove(tmp_path)

#         all_chunks = []
#         for p in pages:
#             ch = split_into_chunks(p["text"], p["page"], DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
#             all_chunks.extend(ch)
#         all_chunks = dedupe_chunks(all_chunks)
#         if not all_chunks:
#             return jsonify({"error": "No readable text extracted from PDF."}), 400

#         texts = [c["chunk"] for c in all_chunks]
#         embeddings = get_embeddings(texts)

#         index = build_faiss_index(embeddings)
#         index.add(embeddings)

#         user_indices[user_id] = {
#             "index": index,
#             "embeddings": embeddings,
#             "meta": all_chunks,
#             "dim": embeddings.shape[1],
#         }
#         return jsonify({
#             "message": "Training complete",
#             "chunks": len(all_chunks),
#             "dimension": embeddings.shape[1],
#         })
#     except Exception as e:
#         print("❌ Error in /train:", e)
#         return jsonify({"error": str(e)}), 500


# @app.route("/query", methods=["POST"])
# def query():
#     try:
#         data = request.get_json(force=True)
#         question = data.get("question", "").strip()
#         user_id = data.get("user_id")
#         top_k = int(data.get("top_k", DEFAULT_TOP_K))

#         if not question:
#             return jsonify({"error": "Missing question"}), 400
#         if not user_id or user_id not in user_indices:
#             return jsonify({"error": "User index not found. Upload a PDF first."}), 400

#         store = user_indices[user_id]
#         index = store["index"]
#         meta = store["meta"]

#         q_emb = get_embeddings([question])[0]

#         over_k = max(top_k * 3, top_k)
#         D, I = index.search(q_emb.reshape(1, -1), over_k)
#         cand_indices = I[0].tolist()

#         doc_vecs = store["embeddings"][cand_indices]
#         mmr_idx_local = maximal_marginal_relevance(q_emb, doc_vecs, top_k=top_k, lambda_mult=0.6)
#         selected_global = [cand_indices[i] for i in mmr_idx_local]

#         contexts = [meta[i] for i in selected_global]
#         answer = generate_answer(question, contexts)

#         sources = []
#         for rank, gi in enumerate(selected_global, 1):
#             sources.append({
#                 "rank": rank,
#                 "page": meta[gi]["page"],
#                 "chunk_preview": (meta[gi]["chunk"][:220] + "...") if len(meta[gi]["chunk"]) > 220 else meta[gi]["chunk"]
#             })

#         return jsonify({"answer": answer, "sources": sources})
#     except Exception as e:
#         print("❌ Error in /query:", e)
#         return jsonify({"error": str(e)}), 500


# @app.route("/home")
# def health():
#     return jsonify({"status": "ok"})


# if __name__ == "__main__":
#     app.run(port=5001, debug=True)



from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import tempfile
import os
import shutil

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Config
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in environment variables")

MODEL_NAME = "gemini-1.5-flash-latest"
EMBED_MODEL = "models/embedding-001"
DEFAULT_TOP_K = int(os.getenv("TOP_K", 5))

# Per-user storage
user_chains = {}  # user_id -> RetrievalQA chain
user_vectorstores = {}  # user_id -> Chroma instance
VECTORSTORE_DIR = "vectorstores"
os.makedirs(VECTORSTORE_DIR, exist_ok=True)


def create_qa_chain_from_pdf(pdf_path, persist_dir):
    """Process PDF and return a RetrievalQA chain."""
    # Load PDF pages
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = splitter.split_documents(docs)

    # Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=GOOGLE_API_KEY
    )

    # Vector store (persistent per-user)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)  # clear old
    vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_dir)

    # Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": DEFAULT_TOP_K})

    # Prompt
    prompt_template = PromptTemplate.from_template("""
    Context:
    {context}

    Question: {question}

    Answer the question using the context above.
    Provide sources when relevant.
    Answer:
    """)

    # LLM
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )

    # Build RetrievalQA
    qa = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    return qa, vectorstore


@app.route("/train", methods=["POST"])
def train():
    try:
        file = request.files.get("file")
        user_id = request.form.get("user_id")

        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400
        if not file or not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Please upload a valid PDF file"}), 400

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        persist_dir = os.path.join(VECTORSTORE_DIR, user_id)
        qa_chain, vectorstore = create_qa_chain_from_pdf(tmp_path, persist_dir)

        # Store in memory
        user_chains[user_id] = qa_chain
        user_vectorstores[user_id] = vectorstore

        return jsonify({"message": "Training complete for user", "user_id": user_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if "tmp_path" in locals():
            os.unlink(tmp_path)


@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.get_json(force=True)
        question = data.get("question", "").strip()
        user_id = data.get("user_id")

        if not question:
            return jsonify({"error": "Missing question"}), 400
        if not user_id or user_id not in user_chains:
            return jsonify({"error": "User not trained yet"}), 400

        qa_chain = user_chains[user_id]
        result = qa_chain.invoke({"query": question})

        # Sources
        sources = []
        for doc in result["source_documents"]:
            sources.append({
                "page": doc.metadata.get("page", None),
                "source": doc.metadata.get("source", None),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })

        return jsonify({
            "answer": result["result"],
            "sources": sources
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/home")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(port=5001, debug=True)
