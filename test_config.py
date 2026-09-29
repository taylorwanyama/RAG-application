from app.config import settings

print(f'Pinecone API key:{settings.pinecone_api_key}')
print(f'Pinecone Index: {settings.pinecone_index}')
print(f'Groq API key: {settings.groq_api_key}')
print(f'Groq Model: {settings.groq_model}')
print(f'Groq Timeout: {settings.groq_timeout}')
print(f'Environment: {settings.environment}')