import json
import os

print("START INDEXING")

# chemins CORRECTS
issues_path = "/home/test-pc/Bureau/PFE_Chatbot/PFE_Chatbot_STM32Cube/data/chunks_issues_stm32cubeh7_v2.json"
files_path = "/home/test-pc/Bureau/PFE_Chatbot/PFE_Chatbot_STM32Cube/data/chunks_files_stm32cubeh7_v2.json"

print("Checking paths...")
print("Issues exists:", os.path.exists(issues_path))
print("Files exists:", os.path.exists(files_path))

# charger
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
    print("files loaded:", len(files))
except Exception as e:
    print("Files error:", e)
    exit()

all_chunks = issues + files

print("Total chunks:", len(all_chunks))

if len(all_chunks) == 0:
    print("No data found → STOP")
    exit()

texts = [c.get("text", "") for c in all_chunks]
metadatas = [c.get("metadata", {}) for c in all_chunks]

print("Creating embeddings...")

try:
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma




    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    BATCH_SIZE = 64

    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    def batch_iter(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield i, data[i:i + batch_size]

    for i, batch in batch_iter(all_chunks, BATCH_SIZE):
        try:
            batch_texts = [c.get("text", "") for c in batch]
            batch_meta = [c.get("metadata", {}) for c in batch]

            print("Max chunk size:", max(len(t) for t in batch_texts))

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
    print("DONE! DB created")


except Exception as e:
    print("ERROR during indexing:", e)
