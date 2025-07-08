from llama_index.core import VectorStoreIndex
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings

import sys
from QAWithPDF.exception import customexception
from QAWithPDF.logger import logging

def download_gemini_embedding(model, document):
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings.

    Returns:
    - Query engine for similarity queries.
    """
    try:
        logging.info("Initializing Gemini Embedding model...")
        gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")

        # Set global settings
        Settings.llm = model
        Settings.embed_model = gemini_embed_model
        Settings.chunk_size = 800
        Settings.chunk_overlap = 20

        logging.info("Building vector index...")
        index = VectorStoreIndex.from_documents(document)
        index.storage_context.persist()

        logging.info("Creating query engine...")
        query_engine = index.as_query_engine()
        return query_engine
    except Exception as e:
        logging.error(f"Error in download_gemini_embedding: {e}")
        raise customexception(e, sys)