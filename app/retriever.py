import asyncio


class PineconeRetriever:

    def __init__(self, embedding_model, pinecone_index):
        self.embedding_model = embedding_model
        self.pinecone_index = pinecone_index

    async def retrieve(self, question, top_k=2):
        query_embedding = self.embedding_model.encode(question)

        result = await asyncio.wait_for(
            asyncio.to_thread(
                self.pinecone_index.query,
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            ),
            timeout=10.0
        )

        return result["matches"] 
    


    