import asyncio
import time
import logging

logger = logging.getLogger(__name__)

class PineconeRetriever:

    def __init__(self, embedding_model, pinecone_index):
        self.embedding_model = embedding_model
        self.pinecone_index = pinecone_index

    async def retrieve(self, question, top_k=2):
        start_time = time.perf_counter()

        embedding_start_time = time.perf_counter()

        query_embedding = self.embedding_model.encode(question)

        embedding_duration = time.perf_counter() - embedding_start_time

        logger.info(
            'Embedding completed | duration=%.3fs',
            embedding_duration
        )

        pinecone_start_time = time.perf_counter()

        result = await asyncio.wait_for(
            asyncio.to_thread(
                self.pinecone_index.query,
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            ),
            timeout=10.0
        )

        pinecone_duration = time.perf_counter() - pinecone_start_time

        logger.info(
            'Pinecone retrieval completed | duration=%.3fs',
            pinecone_duration
        )

        total_duration = time.perf_counter() - start_time

        logger.info(
            'Total retrieval | duration=%.3fs',
            total_duration
        )

        return result["matches"] 
    


    