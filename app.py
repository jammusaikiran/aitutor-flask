from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
import uuid

# Load environment variables
load_dotenv()

GEMINI_API_KEY="AIzaSyDARuh7iFtavn5p0D_3-XjKpS9CLI_yiN0"
PINECONE_API_KEY="pcsk_2ygM7G_3bH69GDUiG8HLzAz4XFLMCVp4bfxj6LfEiAZ3Zyn6tU2MpFhy2vgQgjsu7p71X8"
PINECONE_ENVIRONMENT="us-east1-gcp"
PINECONE_INDEX="pdf-qa1"

if not all([GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX]):
    raise ValueError("Missing API keys or Pinecone config in .env")

# Init Pinecone v3
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Init HuggingFace embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-albert-small-v2")

# Init Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

app = Flask(__name__)
CORS(app)

# Helper: Extract text from PDF
def extract_text_from_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in pdf_document:
        text += page.get_text()
    return text

# Helper: Split text into chunks
def split_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# API: Train with PDF
@app.route("/train", methods=["POST"])
def train():
    try:
        pdf_file = request.files["file"]
        user_id = request.form.get("user_id", str(uuid.uuid4()))

        text = extract_text_from_pdf(pdf_file)
        chunks = split_text(text)

        vectors = []
        for i, chunk in enumerate(chunks):
            vector = embedding_model.embed_query(chunk)
            vectors.append({
                "id": f"{user_id}-{i}",
                "values": vector,
                "metadata": {"text": chunk}
            })

        index.upsert(vectors=vectors, namespace=user_id)

        return jsonify({"status": "success", "chunks": len(chunks), "user_id": user_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API: Query PDF
@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.json
        user_id = data.get("user_id")
        question = data.get("question")

        if not user_id or not question:
            return jsonify({"error": "Missing user_id or question"}), 400

        query_vector = embedding_model.embed_query(question)
        results = index.query(
            vector=query_vector,
            top_k=5,
            namespace=user_id,
            include_metadata=True
        )

        context = "\n".join([match["metadata"]["text"] for match in results["matches"]])
        if not context.strip():
            return jsonify({"answer": "No relevant content found."})

        prompt = f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        response = model.generate_content(prompt)

        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True,port=5001)
