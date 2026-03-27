import json

# charger docs avec similarité
with open("data/docs_issues_v2_sim.json") as f:
    docs = {d["id"]: d for d in json.load(f)}

def enrich(results):
    enriched = []

    for r in results:
        enriched.append(r.page_content)

        related = r.metadata.get("related_issue_ids", [])
        for rid in related[:2]:
            if rid in docs:
                enriched.append(docs[rid]["clean_text"])

    return "\n\n".join(enriched)
