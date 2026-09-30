#import os
import asyncio
#from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import AsyncGroq
from app.config import settings


from app.dependencies import RAGDependencies
#load_dotenv()

#PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
#GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Load the embedding model
#embedding_model = SentenceTransformer("all-MiniLM-L6-v2") remaove it because it now belongs to the lifespan function in main.py


# Connect to Pinecone
# pc = Pinecone(api_key="invalid-key") # Deliberately breaking the system to test error handling
#pc = Pinecone(api_key=PINECONE_API_KEY)
#pc = Pinecone(api_key=settings.pinecone_api_key)
#index = pc.Index("company-book")
#index = pc.Index(settings.pinecone_index)


# Connect to Groq
#groq_client = AsyncGroq(
    #api_key=settings.groq_api_key,
    #timeout=settings.groq_timeout   # Setting a timeout of 30 seconds for Groq API requests
    #)


async def retrieve_chunks(
    question,
    dependencies: RAGDependencies,
    top_k=2):

    query_embedding = dependencies.embedding_model.encode(question)

    result = await asyncio.wait_for(
        asyncio.to_thread(
            dependencies.pinecone_index.query,
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        ),
        timeout=10.0
    )

    return result["matches"]


def build_context(matches):

    retrieved_chunks = []

    for match in matches:
        retrieved_chunks.append(
            match["metadata"]["text"]
        )

    return "\n\n".join(retrieved_chunks)


async def generate_answer(
    question,
    context,
    dependencies: RAGDependencies
    ):

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

    response = await dependencies.groq_client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


async def answer_question(
    question,
    dependencies: RAGDependencies
):

    matches = await retrieve_chunks(
        question,
        dependencies
    )

    context = build_context(matches)

    answer = await generate_answer(
        question,
        context,
        dependencies
    )

    return answer