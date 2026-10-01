from dataclasses import dataclass

@dataclass
class RAGDependencies:
    #embedding_model: object
    #pinecone_index: object  # I have re4moved pinecone specific dependencies from RAG service and putting behind a retriever class.
    retriever: object
    groq_client: object