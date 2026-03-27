from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

def retrieve(query, filters=None):
    results = db.similarity_search(query, k=10)

    if filters:
        results = [
            r for r in results
            if (
                (not filters["component"] or r.metadata.get("component") == filters["component"])
                and
                (not filters["layer"] or r.metadata.get("layer") == filters["layer"])
            )
        ]

    return results[:5]
