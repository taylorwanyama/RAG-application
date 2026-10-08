#import os
import asyncio
#from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import AsyncGroq
from app.config import settings
import logging
import time

from app.dependencies import RAGDependencies
#load_dotenv()

logger = logging.getLogger(__name__)

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
    dependencies,
    top_k=5):
    return await dependencies.retriever.retrieve(
        question,
        top_k=top_k
    )


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
    groq_start_time = time.perf_counter()
 
    response = await dependencies.groq_client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    groq_duration = time.perf_counter() - groq_start_time
    logger.info(
        'Groq response completed | duration=%.3fs',
        groq_duration
    )

    return response.choices[0].message.content


async def answer_question(question,dependencies):
    logger.info('Starting RAG request')

    matches = await retrieve_chunks(
        question,
        dependencies
    )
    logger.info(
        'Retrieved %d relevant chunks',
        len(matches)
    ) 
    context = build_context(matches)

    answer = await generate_answer(
        question,
        context,
        dependencies
    )

    logger.info('RAG request completed.')

    return answer,  [
    {
        "tender_ref": match["metadata"]["tender_ref"],
        "county_id": match["metadata"]["county_id"],
        "procuring_entity": match["metadata"]["procuring_entity"],
        "procurement_method": match["metadata"]["procurement_method"],
        "procurement_category": match["metadata"]["procurement_category"],
        "close_at": match["metadata"]["close_at"],
        "published_at": match["metadata"]["published_at"],
        "venue": match["metadata"]["venue"],
        "tender_fee": match["metadata"]["tender_fee"],
        "validity_in_days": match["metadata"]["validity_in_days"],
        "financial_year": match["metadata"]["financial_year"],
        "tender_document_urls": match["metadata"]["tender_document_urls"]
    }
    for match in matches
]