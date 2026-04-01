import json
import os

from metadata_utils import normalize_metadata_for_chroma

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

print("START INDEXING")


issues_path = "/home/test-pc/Bureau/PFE_Chatbot/PFE_Chatbot_STM32Cube/data/chunks_issues_stm32cubeh7_v2.json"
files_path = "/home/test-pc/Bureau/PFE_Chatbot/PFE_Chatbot_STM32Cube/data/chunks_files_stm32cubeh7_v2.json"

print("Checking paths...")
print("Issues exists:", os.path.exists(issues_path))
print("Files exists:", os.path.exists(files_path))


try:
    with open(issues_path) as f:
        issues = json.load(f)
    print("Issues loaded:", len(issues))
except Exception as e:
    print("Issues error:", e)
    exit()

try:
    with open(files_path) as f:
        files = json.load(f)
    print("Files loaded:", len(files))
except Exception as e:
    print("Files error:", e)
    exit()


all_chunks = issues + files
print("Total chunks:", len(all_chunks))

if len(all_chunks) == 0:
    print("No data found → STOP")
    exit()


BATCH_SIZE = 16
MAX_CHUNK_SIZE = 1000


def batch_iter(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield i, data[i:i + batch_size]


print("Creating embeddings...")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)


for i, batch in batch_iter(all_chunks, BATCH_SIZE):

    try:
        batch_texts = [
            c.get("text", "")[:MAX_CHUNK_SIZE]
            for c in batch
        ]

        batch_meta = [
            normalize_metadata_for_chroma(c.get("metadata", {}))
            for c in batch
        ]

        max_len = max(len(t) for t in batch_texts)
        print("Max chunk size:", max_len)

        db.add_texts(
            texts=batch_texts,
            metadatas=batch_meta
        )

        print(f"Indexed batch {i} → {i + len(batch)}")

    except Exception as e:
        print("Batch error:", e)
        print("Problem batch index:", i)
        break


db.persist()

print("DONE! Chroma DB created successfully ")
