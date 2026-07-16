from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def create_vector_store(chunks):

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        embeddings.astype("float32")
    )

    return index, embeddings

def search_chunks(
    query,
    index,
    chunks,
    top_k=3
):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        query_embedding.astype("float32"),
        top_k
    )

    results = []

    for idx in indices[0]:
        results.append(
            chunks[idx]
        )

    return results