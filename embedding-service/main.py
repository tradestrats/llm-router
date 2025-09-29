"""
Embedding Service - FastAPI service for generating text embeddings
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import time
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model: Optional[SentenceTransformer] = None
model_name: Optional[str] = None

# Concurrency control
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Pydantic models
class EmbedRequest(BaseModel):
    text: str = Field(..., description="Text to embed", min_length=1, max_length=8192)
    
class BatchEmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed", min_length=1, max_length=100)
    
class EmbedResponse(BaseModel):
    embedding: List[float] = Field(..., description="Generated embedding vector")
    dimensions: int = Field(..., description="Number of dimensions in the embedding")
    model_name: str = Field(..., description="Name of the model used")
    
class BatchEmbedResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="List of generated embedding vectors")
    dimensions: int = Field(..., description="Number of dimensions in the embedding")
    model_name: str = Field(..., description="Name of the model used")
    count: int = Field(..., description="Number of embeddings generated")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_name: Optional[str] = Field(None, description="Name of the loaded model")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")

# Startup time tracking
startup_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load model on startup"""
    global model, model_name
    
    logger.info("Starting embedding service...")
    
    # Load the model from cache
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    cache_dir = os.getenv("MODEL_CACHE_DIR", "/app/models")
    logger.info(f"Loading model: {model_name}")
    logger.info(f"From cache directory: {cache_dir}")
    
    try:
        # Load model from cache directory (downloaded at build time)
        model = SentenceTransformer(model_name, cache_folder=cache_dir)
        logger.info(f"Model loaded successfully from cache: {model_name}")
        logger.info(f"Model dimensions: {model.get_sentence_embedding_dimension()}")
    except Exception as e:
        logger.error(f"Failed to load model from cache: {e}")
        # Fallback: try to download if cache fails
        logger.info("Attempting to download model as fallback...")
        try:
            model = SentenceTransformer(model_name)
            logger.info(f"Model downloaded and loaded successfully: {model_name}")
            logger.info(f"Model dimensions: {model.get_sentence_embedding_dimension()}")
        except Exception as fallback_error:
            logger.error(f"Fallback model loading also failed: {fallback_error}")
            raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down embedding service...")

# Create FastAPI app
app = FastAPI(
    title="Embedding Service",
    description="FastAPI service for generating text embeddings using sentence transformers",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - startup_time
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_name=model_name,
        uptime_seconds=uptime
    )

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}

@app.get("/live")
async def liveness_check():
    """Liveness check endpoint"""
    return {"status": "alive"}

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """Generate embedding for a single text"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Acquire semaphore to limit concurrent requests
    async with request_semaphore:
        try:
            # Generate embedding
            embedding = model.encode(request.text, convert_to_tensor=False)
            
            return EmbedResponse(
                embedding=embedding.tolist(),
                dimensions=len(embedding),
                model_name=model_name
            )
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {str(e)}")

@app.post("/embed/batch", response_model=BatchEmbedResponse)
async def embed_batch(request: BatchEmbedRequest):
    """Generate embeddings for multiple texts"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Acquire semaphore to limit concurrent requests
    async with request_semaphore:
        try:
            # Generate embeddings
            embeddings = model.encode(request.texts, convert_to_tensor=False)
            
            return BatchEmbedResponse(
                embeddings=[emb.tolist() for emb in embeddings],
                dimensions=embeddings.shape[1],
                model_name=model_name,
                count=len(embeddings)
            )
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate batch embeddings: {str(e)}")

@app.get("/info")
async def get_info():
    """Get service information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": model_name,
        "dimensions": model.get_sentence_embedding_dimension(),
        "max_text_length": 8192,
        "version": "1.0.0"
    }

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    workers = int(os.getenv("WORKERS", "5"))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info"
    )
