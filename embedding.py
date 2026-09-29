from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = [
    "Employees receive 21 days of annual leave each year.",
    "Employees must submit leave requests two weeks in advance.",
    "The company provides medical insurance to all employees."
]

embeddings = model.encode(chunks)

# print(embeddings.shape)

query = "How many days of annual leave do employees get?"

query_embedding = model.encode(query)

# print(query_embedding.shape)

similarities = cosine_similarity(
    query_embedding.reshape(1, -1),
    embeddings
)

print(similarities)

for i, score in enumerate(similarities[0]):
    print(f"Chunk {i}: {score:.4f}")

top_k = 2

indices = similarities[0].argsort()[-top_k:][::-1]

#for index in indices:
   # print("Score:", similarities[0][index])
    #print("Chunk:", chunks[index])
    #print("---")    

# Create Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create the index
pc.create_index(
    name="company-book1",
    vector_type="dense",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Connect to the index
index = pc.Index("company-book1")

# Prepare vectors
vectors = []

for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    vectors.append({
        "id": f"chunk-{i}",
        "values": embedding.tolist(),
        "metadata": {
            "text": chunk,
            "source": "handbook.txt",
            "chunk_id": i
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

#print(result)
#To explicits retrieve the context
matches = result['matches']
retrieved_chunks = []

for match in matches:
    retrieved_chunks.append(match['metadata']['text'])

print(retrieved_chunks) 
context = "\n\n".join(retrieved_chunks)

print(context)
# We can replace the index cretion with
#pc = Pinecone(api_key=PINECONE_API_KEY)

#if not pc.has_index("company-book"):
    #pc.create_index(
       # name="company-book",
        #vector_type="dense",
        #dimension=384,
        ##metric="cosine",
        #spec=ServerlessSpec(
            #cloud="aws",
            #region="us-east-1"
       # )
   # )

#index = pc.Index("company-book")
def retrieve_chunks(question, top_k=2):
    query_embedding = model.encode(question)

    result = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    return result["matches"]

# Creating the test cases
test_cases = [
    {
        "question": "How many days of annual leave do employees get?",
        "expected": "Employees receive 21 days of annual leave each year."
    },
    {
        "question": "How far in advance must leave be requested?",
        "expected": "Employees must submit leave requests two weeks in advance."
    },
    {
        "question": "What is the maternity leave policy?",
        "expected": None
    }
]

for test in test_cases:

    question = test["question"]

    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("=" * 60)

    matches = retrieve_chunks(question)

    for rank, match in enumerate(matches, start=1):

        print(f"\nRank: {rank}")
        print(f"Score: {match['score']:.4f}")
        print(f"Text: {match['metadata']['text']}")

    print("\nExpected answer:")
    print(test["expected"])

    