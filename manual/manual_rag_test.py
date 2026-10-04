from app.rag_service import answer_question

question = 'How many days of annual leave do employees get?' 
answer = answer_question(question)

print(f'Answer: {answer}')