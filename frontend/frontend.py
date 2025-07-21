import streamlit as st
import requests
import pandas as pd
import os
import json
import time
import zipfile
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# ----------- CONFIG ----------- #
BACKEND_URL = "http://localhost:8000"
LOG_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "vector_db", "document_log.json"))
st.set_page_config(page_title="AgenticRAG Auto Upload", layout="wide")
st.title("📄 AgenticRAG - Auto Upload & Visualization")

# ----------- STATE ----------- #
if "show_bar_modal" not in st.session_state:
    st.session_state["show_bar_modal"] = False
if "show_heatmap_modal" not in st.session_state:
    st.session_state["show_heatmap_modal"] = False
if "show_keyword_modal" not in st.session_state:
    st.session_state["show_keyword_modal"] = False
if "show_pca_modal" not in st.session_state:
    st.session_state["show_pca_modal"] = False
if "uploaded" not in st.session_state:
    st.session_state["uploaded"] = False
if "table_data" not in st.session_state:
    st.session_state["table_data"] = []

# ----------- HELPERS ----------- #
def get_uploaded_filenames():
    try:
        if os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, "r") as f:
                log_data = json.load(f)
                return [entry["filename"] for entry in log_data]
    except Exception as e:
        st.warning(f"⚠️ Could not read log: {e}")
    return []

def simulate_progress(start, end, label, speed=0.03):
    for i in range(start, end + 1, 5):
        progress_bar.progress(i / 100.0)
        status_text.info(f"{label}... {i}%")
        time.sleep(speed)

# ----------- BACKEND STATUS ----------- #
st.subheader("🛠️ Backend Status")
try:
    resp = requests.get(f"{BACKEND_URL}/status", timeout=3)
    if resp.status_code == 200:
        status_data = resp.json()
        st.success("✅ Backend is running")
        st.info(f"Total Vectors Stored: {status_data.get('total_vectors_stored', 0)}")
        st.caption(f"Database Path: {status_data.get('database_path', 'Unknown')}")
    else:
        st.error("❌ Backend did not respond as expected.")
except Exception as e:
    st.error("❌ Could not connect to backend.")

st.divider()

# ----------- UPLOAD SECTION ----------- #
st.header("📤 Upload Documents or Zipped Folder (Auto Process)")
upload = st.file_uploader(
    "Upload documents (PDF, DOCX, TXT, CSV) or a ZIP folder:",
    type=["pdf", "docx", "txt", "csv", "zip"],
    accept_multiple_files=True
)

progress_bar = st.progress(0)
status_text = st.empty()

if upload:
    uploaded_filenames = get_uploaded_filenames()
    table_data = []
    for file in upload:
        if file.name in uploaded_filenames:
            st.warning(f"🚫 {file.name} has already been uploaded. Skipping.")
            continue

        with st.spinner(f"Processing: {file.name}..."):
            files = {"file": (file.name, file.getvalue())}
            simulate_progress(0, 40, "Uploading document")
            try:
                resp = requests.post(f"{BACKEND_URL}/upload", files=files)
                simulate_progress(40, 90, "Processing document")
                if resp.status_code == 200:
                    result = resp.json()
                    simulate_progress(90, 100, "Finalizing")
                    st.success(f"✅ {file.name} processed successfully.")

                    # Document Summary
                    st.subheader(f"📄 Results for: {file.name}")
                    st.write(f"**Total Chunks Indexed:** {result.get('chunks_stored', 0)}")
                    st.write(f"**Embedding Dimension:** {result.get('embedding_dimension', 0)}")

                    # Embedding Preview
                    embedding_sample = result.get("embedding_sample", [])
                    if embedding_sample:
                        st.write("🔍 **Sample Embedding Vector (first 5 dimensions):**")
                        st.code(embedding_sample)

                    # Document Preview
                    st.write("📑 **Preview of Indexed Chunks:**")
                    preview_chunks = result.get("preview_chunks", [])
                    for idx, chunk in enumerate(preview_chunks, 5):
                        st.write(f"**Chunk {idx}:** {chunk[:300]}...")

                    # Append to Table Data
                    table_data.append({
                        "Document": file.name,
                        "Chunks": result.get("chunks_stored", 0),
                        "Embedding Dimension": result.get("embedding_dimension", 0),
                        "Embedding": result.get("embedding_preview", [])
                    })
                else:
                    st.error(f"❌ Failed to process {file.name}. Error: {resp.text}")
            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {str(e)}")

    # Update session state
    if table_data:
        st.session_state["table_data"] = table_data
        st.session_state["uploaded"] = True

st.divider()

# ----------- VISUALIZATION CONTROLS ----------- #
if st.session_state["table_data"]:
    df = pd.DataFrame(st.session_state["table_data"])
    st.subheader("📊 Upload Summary Table")
    st.dataframe(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📈 Generate Bar Graph"):
            st.session_state["show_bar_modal"] = True
    with col2:
        if st.button("🔥 Generate Embedding Heatmap"):
            st.session_state["show_heatmap_modal"] = True
    with col3:
        if st.button("🧠 Keyword Frequency Chart"):
            st.session_state["show_keyword_modal"] = True
    with col4:
        if st.button("🌐 Generate Embedding PCA Plot"):
            st.session_state["show_pca_modal"] = True

# ----------- EXPANDERS ----------- #
if st.session_state["show_bar_modal"]:
    with st.expander("📈 Chunks per Document", expanded=True):
        fig, ax = plt.subplots()
        ax.bar(df["Document"], df["Chunks"], color="skyblue")
        ax.set_xlabel("Document")
        ax.set_ylabel("Chunks Stored")
        ax.set_title("Chunks per Document")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

if st.session_state["show_heatmap_modal"]:
    with st.expander("🔥 Embedding Similarity Heatmap", expanded=True):
        for row in st.session_state["table_data"][:5]:
            doc = row["Document"]
            embeddings = row.get("Embedding", [])

            if embeddings and len(embeddings) > 1:
                st.markdown(f"**{doc}**")
                emb_array = np.array(embeddings)
                norm = np.linalg.norm(emb_array, axis=1, keepdims=True)
                similarity_matrix = np.round(np.dot(emb_array, emb_array.T) / (norm @ norm.T), 2)

                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(similarity_matrix, cmap="coolwarm", annot=False, xticklabels=False, yticklabels=False, ax=ax)
                ax.set_title(f"Semantic Similarity (Cosine) between chunks - {doc}")
                st.pyplot(fig)
            else:
                st.warning(f"Not enough chunks to show similarity heatmap for: {doc}")

if st.session_state["show_keyword_modal"]:
    with st.expander("🧠 Top Keywords per Document", expanded=True):
        for row in st.session_state["table_data"]:
            doc = row["Document"]
            chunks = row.get("Embedding", [])
            if not chunks:
                continue
            words = " ".join([str(c) for c in chunks])
            words = [w.lower() for w in words.split() if w.isalpha() and len(w) > 3]
            freq = Counter(words).most_common(10)
            if not freq:
                continue
            keywords, counts = zip(*freq)

            fig, ax = plt.subplots()
            ax.barh(keywords, counts, color="mediumseagreen")
            ax.set_title(f"Top Keywords: {doc}")
            ax.invert_yaxis()
            st.pyplot(fig)

if st.session_state["show_pca_modal"]:
    with st.expander("🌐 2D Embedding Projection (PCA)", expanded=True):
        for row in st.session_state["table_data"]:
            doc = row["Document"]
            embeddings = row.get("Embedding", [])
            if len(embeddings) < 2:
                continue
            X = normalize(np.array(embeddings))
            pca = PCA(n_components=2)
            proj = pca.fit_transform(X)

            fig, ax = plt.subplots()
            ax.scatter(proj[:, 0], proj[:, 1], alpha=0.7)
            ax.set_title(f"PCA 2D View: {doc}")
            st.pyplot(fig)


# ----------- HISTORY ----------- #
st.header("📚 Document Upload History")
if os.path.exists(LOG_FILE_PATH):
    try:
        with open(LOG_FILE_PATH, "r") as f:
            log_data = json.load(f)

        if log_data:
            st.write("### Uploaded Documents:")
            cols = st.columns([3, 2, 3, 2])
            cols[0].write("**Filename**")
            cols[1].write("**Chunks**")
            cols[2].write("**Uploaded At**")
            cols[3].write("**Action**")

            for idx, entry in enumerate(sorted(log_data, key=lambda x: x["upload_time"], reverse=True)):
                cols = st.columns([3, 2, 3, 2])
                cols[0].write(entry["filename"])
                cols[1].write(entry["chunks_stored"])
                cols[2].write(entry["upload_time"])

                if cols[3].button("❌ Delete", key=f"delete_{idx}_{entry['filename']}"):
                    try:
                        delete_resp = requests.delete(
                            f"{BACKEND_URL}/delete-document",
                            params={"filename": entry['filename']}
                        )
                        if delete_resp.status_code == 200:
                            st.success(f"{entry['filename']} deleted. Refreshing...")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {entry['filename']}: {delete_resp.text}")
                    except Exception as e:
                        st.error(f"Error deleting {entry['filename']}: {e}")
        else:
            st.info("ℹ️ No uploads logged yet.")
    except Exception as e:
        st.warning("⚠️ Couldn't load document log.")
else:
    st.caption("Log file not found. History won't be displayed.")
