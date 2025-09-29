#!/usr/bin/env python3
"""
Model download script for embedding service
Downloads the sentence transformer model at build time to avoid runtime delays
"""

import os
import sys
import logging
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model():
    """Download the embedding model to cache it locally"""
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    cache_dir = os.getenv("MODEL_CACHE_DIR", "/app/models")
    
    logger.info(f"Downloading model: {model_name}")
    logger.info(f"Cache directory: {cache_dir}")
    
    try:
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Download and cache the model
        model = SentenceTransformer(model_name, cache_folder=cache_dir)
        
        # Test the model to ensure it's working
        test_embedding = model.encode("test sentence")
        logger.info(f"Model downloaded successfully: {model_name}")
        logger.info(f"Model dimensions: {model.get_sentence_embedding_dimension()}")
        logger.info(f"Test embedding shape: {test_embedding.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting model download...")
    success = download_model()
    
    if success:
        logger.info("Model download completed successfully")
        sys.exit(0)
    else:
        logger.error("Model download failed")
        sys.exit(1)
