from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model (free, local)
model = SentenceTransformer("all-MiniLM-L6-v2")

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

    return index, embeddings

def semantic_search(query, index, chunks, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append(chunks[idx])

    return results

def main():
    pra_text = load_text(Path("data/pra_rules.txt"))
    corep_text = load_text(Path("data/corep_instructions.txt"))

    pra_chunks = split_into_chunks(pra_text, "PRA Rulebook")
    corep_chunks = split_into_chunks(corep_text, "COREP Instructions")

    all_chunks = pra_chunks + corep_chunks

    index, _ = build_vector_index(all_chunks)

    user_query = "The bank issued new ordinary shares this quarter"

    retrieved = semantic_search(user_query, index, all_chunks)

    print("\nUser Query:")
    print(user_query)

    print("\nRetrieved Regulatory Context:")
    print("-" * 40)
    for r in retrieved:
        print(r["text"])
        print("Source:", r["source"])
        print()

if __name__ == "__main__":
    main()
