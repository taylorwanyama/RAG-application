import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = SentenceTransformer("all-MiniLM-L6-v2")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("company-book1")

groq_client = Groq(api_key=GROQ_API_KEY)

# Notice wew're not creating a pinecone index here, as it was already created in embedding.py. we can directly use it to query the embeddings.

def retrieve_chunks(question, top_k=2):

    query_embedding = model.encode(question)

    result = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    return result["matches"]

# The context
def build_context(matches):

    retrieved_chunks = []

    for match in matches:
        retrieved_chunks.append(
            match["metadata"]["text"]
        )

    return "\n\n".join(retrieved_chunks)

# Connecting Groq
def generate_answer(question, context):

    prompt = f"""
You are an assistant answering questions using company documents.

Use ONLY the information provided in the context.

If the context does not contain enough information to answer
the question, say that the information is not available in
the provided documents.

Do not use outside knowledge.
Do not invent information.

Context:
{context}

Question:
{question}

Answer:
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    print('\n--- PROMPT SENT TO LLM ---')
    print(prompt)
    print('--- END PROMPT ---')

    return response.choices[0].message.content

# Putting the entire pipeline together
# question = "How many days of annual leave do employees get?"  # This is an answerable question 
question = "What is the company's maternity leave policy?"   # An unanswerable question

matches = retrieve_chunks(question)

context = build_context(matches)

answer = generate_answer(
    question,
    context
)

print("\nRetrieved context:")
print(context)

print("\nAnswer:")
print(answer)