from dataclasses import dataclass

@dataclass
class RAGDependencies:
    embedding_model: object
    pinecone_index: object
    groq_client: object