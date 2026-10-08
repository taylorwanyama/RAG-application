from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from app.config import settings
import json

load_dotenv()

PINECONE_API_KEY = settings.pinecone_api_key

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = json.load(open("data/processed/tender_documents.json"))
texts = [doc["text"] for doc in documents]
embeddings = model.encode(texts)

query = "Which tenders are related to medical equipment?"
#query = "Find tenders involving construction services."

query_embedding = model.encode(query)

similarities = cosine_similarity(
    query_embedding.reshape(1, -1),
    embeddings
)

#print(similarities)

#for i, score in enumerate(similarities[0]):
   # print(f"Chunk {i}: {score:.4f}")

top_k = 2

indices = similarities[0].argsort()[-top_k:][::-1]

pc = Pinecone(api_key=PINECONE_API_KEY)
if not pc.has_index("kenyan-active-tenders"):
    pc.create_index(
        name="kenyan-active-tenders",
        vector_type="dense",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
       )
    )

index = pc.Index("kenyan-active-tenders")


# Prepare vectors
vectors = []

for document, embedding in zip(documents, embeddings):
    vectors.append({
        "id": f"tender_{document['metadata']['id']}",
        "values": embedding.tolist(),
        "metadata": {
            **document["metadata"],
            "text": document["text"]
        }
    })

# Upload vectors
index.upsert(vectors=vectors)

# Query
result = index.query(
    vector=query_embedding.tolist(),
    top_k=2,
    include_metadata=True
)

#To explicits retrieve the context
matches = result['matches']
retrieved_chunks = []

for match in matches:
    retrieved_chunks.append(match['metadata']['text'])

print(retrieved_chunks) 
context = "\n\n".join(retrieved_chunks)

# print(context)
def retrieve_chunks(question, top_k=2):
    query_embedding = model.encode(question)

    result = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    return result["matches"]

for match in result["matches"]:
    print(f"Score: {match['score']:.4f}")
    print(match["metadata"]["text"])    