import json
import os
from typing import List, Dict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from metadata_utils import normalize_metadata_for_chroma

# --------------------
# CONFIG
# --------------------
ISSUES_PATH = "/home/test-pc/Bureau/PFE_Chatbot/PFE_Chatbot_STM32Cube/data/chunks_issues_stm32cubeh7_v2.json"
FILES_PATH = "/home/test-pc/Bureau/PFE_Chatbot/PFE_Chatbot_STM32Cube/data/chunks_files_stm32cubeh7_v2.json"

PERSIST_DIR = "./chroma_db_hf"  # répertoire pour cette version
MAX_CHUNK_SIZE = 1000  # limite de texte pour les embeddings


def load_json(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=== START INDEXING (HuggingFaceEmbeddings) ===")

    print("Checking paths...")
    print("Issues exists:", os.path.exists(ISSUES_PATH))
    print("Files exists:", os.path.exists(FILES_PATH))

    try:
        issues = load_json(ISSUES_PATH)
        print("Issues loaded:", len(issues))
    except Exception as e:
        print("Issues error:", e)
        return

    try:
        files = load_json(FILES_PATH)
        print("Files loaded:", len(files))
    except Exception as e:
        print("Files error:", e)
        return

    all_chunks = issues + files
    print("Total chunks:", len(all_chunks))

    if not all_chunks:
        print("No data found → STOP")
        return

    # --------------------
    # Préparation textes + metadata
    # --------------------
    print("Preparing texts and metadata...")

    texts = [
        (c.get("text") or "")[:MAX_CHUNK_SIZE]
        for c in all_chunks
    ]

    metadatas = [
        normalize_metadata_for_chroma(c.get("metadata", {}))
        for c in all_chunks
    ]

    # Sanity check
    max_len = max(len(t) for t in texts)
    avg_len = sum(len(t) for t in texts) / len(texts)
    print(f"Max chunk length: {max_len}")
    print(f"Avg chunk length: {avg_len:.1f}")

    # --------------------
    # Embeddings HF
    # --------------------
    print("Creating HuggingFace embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --------------------
    # Chroma DB
    # --------------------
    print("Creating Chroma DB and adding texts...")
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )

    db.add_texts(texts=texts, metadatas=metadatas)

    db.persist()
    print("DONE! Chroma DB (HF) created successfully at", PERSIST_DIR)


if __name__ == "__main__":
    main()
    
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="./chroma_db_hf",
    embedding_function=embeddings,
)

query = "Ethernet issue on NUCLEO-H743ZI"
docs = db.similarity_search(query, k=5)
for d in docs:
    print("----")
    print(d.metadata)
    print(d.page_content[:300])