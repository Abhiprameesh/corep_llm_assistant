from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

POLICY_FIELDS = {
    "F1": "Common Equity Tier 1 Capital",
    "F2": "Tier 1 Capital",
    "F3": "Total Capital",
    "F4": "Risk Weighted Assets"
}

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def split_into_chunks(text, source):
    chunks = []
    current = ""

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if ":" in line and current:
            chunks.append({
                "text": current.strip(),
                "source": source
            })
            current = line
        else:
            current += " " + line

    if current:
        chunks.append({
            "text": current.strip(),
            "source": source
        })

    return chunks

def build_vector_index(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index

def semantic_search(query, index, chunks, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    return [chunks[i] for i in indices[0]]

def reason_and_map(query, retrieved_chunks):
    decisions = {}
    audit_log = []

    for chunk in retrieved_chunks:
        text = chunk["text"].lower()

        if "ordinary share" in text or "cet1" in text:
            decisions["F1"] = {
                "impact": "increase",
                "reason": "Ordinary shares contribute to CET1 capital"
            }
            decisions["F2"] = {
                "impact": "increase",
                "reason": "Tier 1 capital includes CET1 capital"
            }
            decisions["F3"] = {
                "impact": "increase",
                "reason": "Total capital includes Tier 1 capital"
            }

            audit_log.append(chunk)

    return decisions, audit_log


def main():
    pra_text = load_text(Path("data/pra_rules.txt"))
    corep_text = load_text(Path("data/corep_instructions.txt"))

    pra_chunks = split_into_chunks(pra_text, "PRA Rulebook")
    corep_chunks = split_into_chunks(corep_text, "COREP Instructions")

    all_chunks = pra_chunks + corep_chunks
    index = build_vector_index(all_chunks)

    user_query = "The bank issued new ordinary shares this quarter"

    retrieved = semantic_search(user_query, index, all_chunks)
    decisions, audit_log = reason_and_map(user_query, retrieved)

    print("\nUser Query:")
    print(user_query)

    print("\nAI Policy Mapping Output:")
    print("-" * 40)
    
    for field, info in decisions.items():
        print(f"Field {field} ({POLICY_FIELDS[field]})")
        print("Impact:", info["impact"])
        print("Reason:", info["reason"])
        print()


    print("Audit Log:")
    print("-" * 40)
    for a in audit_log:
        print(a["text"])
        print("Source:", a["source"])
        print()

if __name__ == "__main__":
    main()
