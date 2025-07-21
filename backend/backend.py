import os
import uuid
import tempfile
import datetime
import json
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document
import chromadb
from chromadb.config import Settings

# Initialize FastAPI
app = FastAPI()

# ---------- GLOBAL PERSISTENT STORAGE PATHS --------------

DB_BASE_DIR = os.path.abspath("./vector_db")
CHROMA_DB_DIR = os.path.join(DB_BASE_DIR, "chroma_db")
LOG_FILE_PATH = os.path.join(DB_BASE_DIR, "document_log.json")

# Ensure required directories exist
os.makedirs(DB_BASE_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# ---------- Initialize Persistent ChromaDB ---------------

from chromadb import PersistentClient

client = PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_or_create_collection("documents")


# ---------- Load Embedding Model -------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# ------------------- MODULES ----------------------

def parse_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n\n".join([page.extract_text() or '' for page in reader.pages])

def parse_docx(file_path):
    doc = Document(file_path)
    return "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def parse_txt_csv(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def split_into_paragraphs(text):
    return [p.strip() for p in text.split('\n\n') if p.strip()]

def log_upload(filename, chunks_count):
    log_data = []
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, "r") as f:
            log_data = json.load(f)

    log_data.append({
        "filename": filename,
        "chunks_stored": chunks_count,
        "upload_time": str(datetime.datetime.now())
    })

    with open(LOG_FILE_PATH, "w") as f:
        json.dump(log_data, f, indent=4)


# ------------------- ENDPOINTS ----------------------

@app.post("/upload")
async def upload(file: UploadFile):
    try:
        ext = file.filename.split(".")[-1].lower()
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file.filename)

        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Parse File
        if ext == "pdf":
            text = parse_pdf(temp_path)
        elif ext == "docx":
            text = parse_docx(temp_path)
        elif ext in ["txt", "csv"]:
            text = parse_txt_csv(temp_path)
        else:
            raise HTTPException(status_code=415, detail="Unsupported file type.")

        # Chunk by Paragraph
        paragraphs = split_into_paragraphs(text)
        if not paragraphs:
            raise HTTPException(status_code=422, detail="No readable text found in document.")

        # Generate Embeddings
        embeddings = model.encode(paragraphs).tolist()

        # Store in ChromaDB
        collection.add(
            ids=[str(uuid.uuid4()) for _ in paragraphs],
            embeddings=embeddings,
            documents=paragraphs,
            metadatas=[{"filename": file.filename}] * len(paragraphs)
        )

        os.remove(temp_path)
        log_upload(file.filename, len(paragraphs))

        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "chunks_stored": len(paragraphs),
            "embedding_dimension": len(embeddings[0]),
            "embedding_sample": embeddings[0][:5],
            "preview_chunks": paragraphs[:3]
        }, 200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def status():
    count = collection.count()
    return {
        "total_vectors_stored": count,
        "database_path": CHROMA_DB_DIR
        
    }
@app.delete("/delete-document")
async def delete_document(filename: str):
    try:
        # Fetch only vector IDs linked to filename
        results = collection.get(where={"filename": filename})

        if results and results["ids"]:
            collection.delete(ids=results["ids"])

        # Efficient JSON log file update
        if os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, "r") as f:
                log_data = json.load(f)

            log_data = [entry for entry in log_data if entry["filename"] != filename]

            with open(LOG_FILE_PATH, "w") as f:
                json.dump(log_data, f, indent=4)

        return {"status": "deleted", "filename": filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


