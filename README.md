# AgenticRAG

backend :<br />
pip install fastapi <br />
pip install uvicorn <br />
pip install sentence-transformers <br />
pip install chromadb <br />
pip install PyPDF2 <br />
pip install python-docx <br />
pip install python-multipart <br />
<br />
frontend : <br />
pip install streamlit <br />
pip install pandas <br />
pip install matplotlib <br />
pip install seaborn <br />
pip install scikit-learn <br />
pip install numpy <br />
requests



# 📄 Document Ingestion API with Embedding and Storage

This FastAPI-based backend service handles document uploads, processes them using NLP embeddings, and stores them in a vector database (ChromaDB) for further analysis or retrieval tasks.

---

## 🚀 Workflow Overview

1. **Receive File Upload**
   - Accepts `.pdf`, `.docx`, and `.txt` files via HTTP POST.

2. **Temporary File Save**
   - The uploaded file is saved in a temporary directory for processing.

3. **Content Parsing**
   - The file is parsed based on its extension:
     - `.pdf`: Extracted using `PyPDF2`
     - `.docx`: Extracted using `python-docx`
     - `.txt`: Read as plain text

4. **Content Splitting**
   - The extracted content is split into paragraphs for chunked processing.

5. **Embedding Generation**
   - Each paragraph is embedded using the `SentenceTransformer` model from Hugging Face.

6. **Vector Storage**
   - Embeddings and corresponding paragraphs are stored in a **ChromaDB** collection.

7. **File Logging**
   - Upload and processing metadata is logged for monitoring or auditing.

8. **JSON Response**
   - On successful processing, the API responds with:
     - `status`: ✅ Success or ❌ Error
     - `filename`: Name of the uploaded file
     - `number_of_chunks`: Total number of paragraphs extracted
     - `embedding_dimension`: Dimensionality of vector embeddings
     - `sample_embedding_values`: Preview of the first paragraph’s vector (first 5 values)
     - `paragraph_preview`: First 3 paragraphs of the document

---

## 📦 Technologies Used

- **FastAPI** – High-performance web framework
- **SentenceTransformer** – Embedding generator from Hugging Face
- **ChromaDB** – Vector database for semantic search
- **PyPDF2** – PDF parsing
- **python-docx** – DOCX parsing
- **tempfile** – For temporary file handling
- **uuid / datetime / os / json** – Utilities for file tracking and management

---

## 📁 Example Response

```json
{
  "status": "success",
  "filename": "sample.pdf",
  "number_of_chunks": 12,
  "embedding_dimension": 384,
  "sample_embedding_values": [0.123, -0.456, 0.789, ..., 0.101],
  "paragraph_preview": [
    "This document discusses the current trends in AI.",
    "Machine Learning and Deep Learning are key components of AI.",
    "Recent advancements include transformers and large language models."
  ]
}
```
# 📄 Document Processing and Vector Storage Service

## 🔄 Example Flow

- You upload `report.pdf`.  
- It gets stored in `/tmp/report.pdf`.  
- Extracts paragraphs → say **20**.  
- **20 embeddings** are generated → each of size **384**.  
- Stored in **ChromaDB** with metadata:  
  ```json
  {
    "filename": "report.pdf"
  }
  ```
- Log entry
  ```json
{
  "filename": "report.pdf",
  "chunks_stored": 20,
  "upload_time": "2025-07-21T10:25:00"
}
```
