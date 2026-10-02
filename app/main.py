from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from pydantic import BaseModel, Field
from app.rag_service import answer_question
import asyncio
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import AsyncGroq
import logging
import time

from app.config import settings
from app.dependencies import RAGDependencies
from app.retriever import PineconeRetriever
from app.rate_limiter import RateLimiter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
#To turn down http library logging noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

rate_limiter = RateLimiter(
    max_requests=3,
    window_seconds=60
)

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info('Starting application')

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
    retriever = PineconeRetriever(
        embedding_model=embedding_model,
        pinecone_index=pinecone_index
   )

    app.state.rag_dependencies = RAGDependencies(
        #embedding_model=embedding_model,
        #pinecone_index=pinecone_index,
        retriever=retriever,
        groq_client=groq_client
    )

    logger.info('Application resources initialized.')

    yield

    logger.info('Shutting down application')


app = FastAPI(lifespan=lifespan)

class QuestionRequest(BaseModel):
    question : str = Field(
        min_length=1,
        max_length=1000
    )

def get_rag_dependencies(request: Request) -> RAGDependencies:
    return request.app.state.rag_dependencies 

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail='Invalid or missing API key.'
        )

def check_rate_limit(
    x_api_key: str = Header(...)
):
    if not rate_limiter.allow(x_api_key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )    
    
@app.get('/health/live') 
async def liveness():
    return {'status': 'Alive'}

@app.get('/health/ready')
async def readiness(request: Request):
    if getattr(request.app.state, 'rag_dependencies', None) is None:
        raise HTTPException(
            status_code=503,
            detail='Applictaion is not ready.'
        )
    return {'status': 'Ready'}

@app.post('/ask')
async def ask(
    question_request: QuestionRequest,
    dependencies: RAGDependencies = Depends(get_rag_dependencies),
    _:None = Depends(verify_api_key),
    __: None = Depends(check_rate_limit)
    ):
    start_time = time.perf_counter()

    try:
        answer = await answer_question(
            question_request.question,
            dependencies
        )
        duration = time.perf_counter() - start_time

        logger.info(
            "RAG request completed | duration=%.3fs",
            duration
        )

        return {
                'Answer': answer
            }
    except asyncio.TimeoutError:
        duration = time.perf_counter() - start_time
        logger.warning(
            "RAG request timed out | duration=%.3fs",
            duration
        )
        raise HTTPException(
            status_code=504,
            detail='A required service took too long to respond. Please try again later.'
        )
    except Exception as e:
        duration = time.perf_counter() - start_time

        #print(f"ERROR: {type(e).__name__}: {e}")
        logger.exception(
            'Unexpected error while processing request| duration=%.3fs',
            duration
        )

        raise HTTPException(
            status_code=500,
            detail='An internal error occurred while processing the request.' 
        )

    