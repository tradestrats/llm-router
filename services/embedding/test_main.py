"""
Tests for the embedding service
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

from main import app

# Test client
client = TestClient(app)

class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_check_unhealthy(self):
        """Test health check when model is not loaded"""
        with patch('main.model', None):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False
    
    def test_health_check_healthy(self):
        """Test health check when model is loaded"""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        
        with patch('main.model', mock_model):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True
    
    def test_readiness_check_unhealthy(self):
        """Test readiness check when model is not loaded"""
        with patch('main.model', None):
            response = client.get("/ready")
            assert response.status_code == 503
    
    def test_readiness_check_healthy(self):
        """Test readiness check when model is loaded"""
        mock_model = MagicMock()
        
        with patch('main.model', mock_model):
            response = client.get("/ready")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
    
    def test_liveness_check(self):
        """Test liveness check"""
        response = client.get("/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

class TestEmbedEndpoints:
    """Test embedding endpoints"""
    
    def test_embed_text_success(self):
        """Test successful text embedding"""
        mock_model = MagicMock()
        mock_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        mock_model.encode.return_value = mock_embedding
        mock_model.get_sentence_embedding_dimension.return_value = 4
        
        with patch('main.model', mock_model):
            response = client.post("/embed", json={"text": "Hello world"})
            assert response.status_code == 200
            data = response.json()
            assert "embedding" in data
            assert len(data["embedding"]) == 4
            assert data["dimensions"] == 4
    
    def test_embed_text_model_not_loaded(self):
        """Test embedding when model is not loaded"""
        with patch('main.model', None):
            response = client.post("/embed", json={"text": "Hello world"})
            assert response.status_code == 503
    
    def test_embed_text_empty_text(self):
        """Test embedding with empty text"""
        mock_model = MagicMock()
        
        with patch('main.model', mock_model):
            response = client.post("/embed", json={"text": ""})
            assert response.status_code == 422  # Validation error
    
    def test_embed_text_too_long(self):
        """Test embedding with text that's too long"""
        mock_model = MagicMock()
        long_text = "a" * 10000  # Longer than max_length
        
        with patch('main.model', mock_model):
            response = client.post("/embed", json={"text": long_text})
            assert response.status_code == 422  # Validation error
    
    def test_embed_batch_success(self):
        """Test successful batch embedding"""
        mock_model = MagicMock()
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 2
        
        with patch('main.model', mock_model):
            response = client.post("/embed/batch", json={"texts": ["Hello", "World"]})
            assert response.status_code == 200
            data = response.json()
            assert "embeddings" in data
            assert len(data["embeddings"]) == 2
            assert data["count"] == 2
            assert data["dimensions"] == 2
    
    def test_embed_batch_model_not_loaded(self):
        """Test batch embedding when model is not loaded"""
        with patch('main.model', None):
            response = client.post("/embed/batch", json={"texts": ["Hello", "World"]})
            assert response.status_code == 503
    
    def test_embed_batch_empty_list(self):
        """Test batch embedding with empty list"""
        mock_model = MagicMock()
        
        with patch('main.model', mock_model):
            response = client.post("/embed/batch", json={"texts": []})
            assert response.status_code == 422  # Validation error
    
    def test_embed_batch_too_many_texts(self):
        """Test batch embedding with too many texts"""
        mock_model = MagicMock()
        many_texts = ["text"] * 101  # More than max_items
        
        with patch('main.model', mock_model):
            response = client.post("/embed/batch", json={"texts": many_texts})
            assert response.status_code == 422  # Validation error

class TestInfoEndpoint:
    """Test info endpoint"""
    
    def test_info_model_not_loaded(self):
        """Test info when model is not loaded"""
        with patch('main.model', None):
            response = client.get("/info")
            assert response.status_code == 503
    
    def test_info_model_loaded(self):
        """Test info when model is loaded"""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        
        with patch('main.model', mock_model):
            response = client.get("/info")
            assert response.status_code == 200
            data = response.json()
            assert "model_name" in data
            assert "dimensions" in data
            assert "max_text_length" in data
            assert "version" in data

class TestErrorHandling:
    """Test error handling"""
    
    def test_embed_model_error(self):
        """Test embedding when model raises an error"""
        mock_model = MagicMock()
        mock_model.encode.side_effect = Exception("Model error")
        
        with patch('main.model', mock_model):
            response = client.post("/embed", json={"text": "Hello world"})
            assert response.status_code == 500
            data = response.json()
            assert "Failed to generate embedding" in data["detail"]
    
    def test_embed_batch_model_error(self):
        """Test batch embedding when model raises an error"""
        mock_model = MagicMock()
        mock_model.encode.side_effect = Exception("Model error")
        
        with patch('main.model', mock_model):
            response = client.post("/embed/batch", json={"texts": ["Hello", "World"]})
            assert response.status_code == 500
            data = response.json()
            assert "Failed to generate batch embeddings" in data["detail"]

if __name__ == "__main__":
    pytest.main([__file__])
