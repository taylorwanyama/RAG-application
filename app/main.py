from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.rag_service import answer_question
import asyncio

app = FastAPI()

class QuestionRequest(BaseModel):
    question : str = Field(
        min_length=1,
        max_length=1000
    )


@app.post('/ask')
async def ask(request: QuestionRequest):

    try:
        answer = await answer_question(request.question)
        return {
                'Answer': answer
            }
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail='A required service took too long to respond. Please try again later.'
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the request." 
        )

    