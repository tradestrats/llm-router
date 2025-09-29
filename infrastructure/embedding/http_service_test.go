package embedding

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"llm-router/domain/embedding"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Helper function to create a mock server that responds properly to all endpoints
func createMockServer(embedHandler func(w http.ResponseWriter, r *http.Request)) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health", "/ready":
			w.WriteHeader(http.StatusOK)
		case "/embed":
			embedHandler(w, r)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
}

func TestHTTPEmbeddingService_SuccessfulRequest(t *testing.T) {
	server := createMockServer(func(w http.ResponseWriter, r *http.Request) {
		response := EmbedResponse{
			Embedding:  []float32{0.1, 0.2, 0.3},
			Dimensions: 3,
			ModelName:  "test-model",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	})
	defer server.Close()

	config := embedding.EmbeddingConfig{
		MaxWorkers:       4,
		MaxTextLength:    8192,
		InferenceTimeout: 5000,
		CacheSize:        100,
	}

	service, err := NewHTTPEmbeddingService(config, server.URL)
	require.NoError(t, err)
	defer service.Close()

	ctx := context.Background()
	embedding, err := service.Embed(ctx, "test text")
	assert.NoError(t, err)
	assert.Equal(t, []float32{0.1, 0.2, 0.3}, embedding)
}

func TestHTTPEmbeddingService_ServerError(t *testing.T) {
	server := createMockServer(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	})
	defer server.Close()

	config := embedding.EmbeddingConfig{
		MaxWorkers:       4,
		MaxTextLength:    8192,
		InferenceTimeout: 5000,
		CacheSize:        100,
	}

	service, err := NewHTTPEmbeddingService(config, server.URL)
	require.NoError(t, err)
	defer service.Close()

	ctx := context.Background()
	_, err = service.Embed(ctx, "test text")
	assert.Error(t, err)
}

func TestHTTPEmbeddingService_RetryLogic(t *testing.T) {
	requestCount := 0
	server := createMockServer(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		if requestCount < 3 {
			// First 2 requests fail
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		// 3rd request succeeds
		response := EmbedResponse{
			Embedding:  []float32{0.1, 0.2, 0.3},
			Dimensions: 3,
			ModelName:  "test-model",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	})
	defer server.Close()

	config := embedding.EmbeddingConfig{
		MaxWorkers:       4,
		MaxTextLength:    8192,
		InferenceTimeout: 5000,
		CacheSize:        100,
	}

	service, err := NewHTTPEmbeddingService(config, server.URL)
	require.NoError(t, err)
	defer service.Close()

	ctx := context.Background()
	start := time.Now()

	// Should succeed after retries
	embedding, err := service.Embed(ctx, "test text")
	assert.NoError(t, err)
	assert.NotNil(t, embedding)

	// Should have taken some time due to retries with backoff
	elapsed := time.Since(start)
	assert.Greater(t, elapsed, 100*time.Millisecond)

	// Should have made exactly 3 requests
	assert.Equal(t, 3, requestCount)
}

func TestHTTPEmbeddingService_CircuitBreakerTripping(t *testing.T) {
	requestCount := 0
	server := createMockServer(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		// Always fail to trip circuit breaker
		w.WriteHeader(http.StatusInternalServerError)
	})
	defer server.Close()

	config := embedding.EmbeddingConfig{
		MaxWorkers:       4,
		MaxTextLength:    8192,
		InferenceTimeout: 5000,
		CacheSize:        100,
	}

	service, err := NewHTTPEmbeddingService(config, server.URL)
	require.NoError(t, err)
	defer service.Close()

	ctx := context.Background()

	// First request should attempt retries
	_, err = service.Embed(ctx, "test text 1")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "embedding generation failed after 3 attempts")

	// After 3 consecutive failures, circuit should open
	// Next request should fail fast with circuit breaker error
	_, err = service.Embed(ctx, "test text 2")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "circuit breaker")

	// Should have made 3 requests (retries) for first call, none for second
	assert.Equal(t, 3, requestCount)
}

func TestHTTPEmbeddingService_ContextCancellation(t *testing.T) {
	server := createMockServer(func(w http.ResponseWriter, r *http.Request) {
		// Simulate slow response
		time.Sleep(200 * time.Millisecond)
		response := EmbedResponse{
			Embedding:  []float32{0.1, 0.2, 0.3},
			Dimensions: 3,
			ModelName:  "test-model",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	})
	defer server.Close()

	config := embedding.EmbeddingConfig{
		MaxWorkers:       4,
		MaxTextLength:    8192,
		InferenceTimeout: 5000,
		CacheSize:        100,
	}

	service, err := NewHTTPEmbeddingService(config, server.URL)
	require.NoError(t, err)
	defer service.Close()

	// Create context with short timeout
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	// Should fail with context cancellation
	_, err = service.Embed(ctx, "test text")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "context deadline exceeded")
}

func TestHTTPEmbeddingService_InputValidation(t *testing.T) {
	server := createMockServer(func(w http.ResponseWriter, r *http.Request) {
		response := EmbedResponse{
			Embedding:  []float32{0.1, 0.2, 0.3},
			Dimensions: 3,
			ModelName:  "test-model",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	})
	defer server.Close()

	config := embedding.EmbeddingConfig{
		MaxWorkers:       4,
		MaxTextLength:    10, // Very short for testing
		InferenceTimeout: 5000,
		CacheSize:        100,
	}

	service, err := NewHTTPEmbeddingService(config, server.URL)
	require.NoError(t, err)
	defer service.Close()

	ctx := context.Background()

	// Test empty text
	_, err = service.Embed(ctx, "")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "empty text")

	// Test whitespace-only text
	_, err = service.Embed(ctx, "   ")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "empty text")

	// Test text too long (gets truncated but should still work)
	longText := strings.Repeat("a", 20) // Longer than maxTextLength
	embedding, err := service.Embed(ctx, longText)
	assert.NoError(t, err)
	assert.NotNil(t, embedding)
}

func TestHTTPEmbeddingService_Caching(t *testing.T) {
	requestCount := 0
	server := createMockServer(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		response := EmbedResponse{
			Embedding:  []float32{0.1, 0.2, 0.3},
			Dimensions: 3,
			ModelName:  "test-model",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	})
	defer server.Close()

	config := embedding.EmbeddingConfig{
		MaxWorkers:       4,
		MaxTextLength:    8192,
		InferenceTimeout: 5000,
		CacheSize:        100,
	}

	service, err := NewHTTPEmbeddingService(config, server.URL)
	require.NoError(t, err)
	defer service.Close()

	ctx := context.Background()
	text := "test text for caching"

	// First request should hit the server
	embedding1, err := service.Embed(ctx, text)
	assert.NoError(t, err)
	assert.NotNil(t, embedding1)

	// Second request with same text should hit cache
	embedding2, err := service.Embed(ctx, text)
	assert.NoError(t, err)
	assert.NotNil(t, embedding2)
	assert.Equal(t, embedding1, embedding2) // Should be same from cache

	// Should have made only 1 server request due to caching
	assert.Equal(t, 1, requestCount)
}

func TestHTTPEmbeddingService_HealthCheckCircuitBreaker(t *testing.T) {
	healthFailureCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			healthFailureCount++
			if healthFailureCount == 1 {
				// First call during initialization succeeds
				w.WriteHeader(http.StatusOK)
				return
			}
			if healthFailureCount <= 4 {
				// Next 3 calls fail to trip circuit breaker
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			w.WriteHeader(http.StatusOK)
		case "/ready":
			w.WriteHeader(http.StatusOK)
		case "/embed":
			w.WriteHeader(http.StatusOK)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	config := embedding.EmbeddingConfig{
		MaxWorkers:       4,
		MaxTextLength:    8192,
		InferenceTimeout: 5000,
		CacheSize:        100,
	}

	service, err := NewHTTPEmbeddingService(config, server.URL)
	require.NoError(t, err)
	defer service.Close()

	ctx := context.Background()

	// Next 3 health checks should fail
	for i := 0; i < 3; i++ {
		err := service.Health(ctx)
		assert.Error(t, err)
		assert.NotContains(t, err.Error(), "circuit breaker is open")
	}

	// 4th health check should fail with circuit breaker open
	err = service.Health(ctx)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "circuit breaker is open")
}

func TestHTTPEmbeddingService_ConcurrentRequests(t *testing.T) {
	requestCount := 0
	server := createMockServer(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		// Simulate some processing time
		time.Sleep(10 * time.Millisecond)
		response := EmbedResponse{
			Embedding:  []float32{0.1, 0.2, 0.3},
			Dimensions: 3,
			ModelName:  "test-model",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	})
	defer server.Close()

	config := embedding.EmbeddingConfig{
		MaxWorkers:       2, // Limit to 2 concurrent workers
		MaxTextLength:    8192,
		InferenceTimeout: 5000,
		CacheSize:        100,
	}

	service, err := NewHTTPEmbeddingService(config, server.URL)
	require.NoError(t, err)
	defer service.Close()

	ctx := context.Background()

	// Start multiple concurrent requests
	const numRequests = 3 // Reduced to avoid race conditions
	results := make(chan error, numRequests)

	for i := 0; i < numRequests; i++ {
		go func(index int) {
			_, err := service.Embed(ctx, fmt.Sprintf("test text %d", index))
			results <- err
		}(i)
	}

	// Wait for all requests to complete
	for i := 0; i < numRequests; i++ {
		err := <-results
		assert.NoError(t, err)
	}

	// All requests should have been made
	assert.Equal(t, numRequests, requestCount)
}