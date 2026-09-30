from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from app.rag_service import answer_question
import asyncio
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import AsyncGroq

from app.config import settings
from app.dependencies import RAGDependencies

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting application...")

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    pinecone = Pinecone(
        api_key=settings.pinecone_api_key
    )

    pinecone_index = pinecone.Index(
        settings.pinecone_index
    )

    groq_client = AsyncGroq(
        api_key=settings.groq_api_key,
        timeout=settings.groq_timeout
    )

    app.state.rag_dependencies = RAGDependencies(
        embedding_model=embedding_model,
        pinecone_index=pinecone_index,
        groq_client=groq_client
    )

    print("Application resources initialized.")

    yield

    print("Shutting down application...")

app = FastAPI(lifespan=lifespan)

class QuestionRequest(BaseModel):
    question : str = Field(
        min_length=1,
        max_length=1000
    )


@app.post('/ask')
async def ask(
    request: Request,
    question_request: QuestionRequest
    ):

    try:
        answer = await answer_question(
            question_request.question,
            request.app.state.rag_dependencies
        )
        return {
                'Answer': answer
            }
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail='A required service took too long to respond. Please try again later.'
        )
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500,
            detail='An internal error occurred while processing the request.' 
        )

    