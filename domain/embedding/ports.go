package embedding

import (
	"context"
	"llm-router/domain/chat"
)

// EmbeddingService defines the interface for text embedding generation
type EmbeddingService interface {
	// Embed generates embeddings for the given text
	Embed(ctx context.Context, text string) ([]float32, error)

	// EmbedMessages generates embeddings for chat messages
	EmbedMessages(ctx context.Context, messages []chat.Message) ([]float32, error)

	// BatchEmbed generates embeddings for multiple texts
	BatchEmbed(ctx context.Context, texts []string) ([][]float32, error)

	// GetDimensions returns the dimensionality of embeddings
	GetDimensions() int

	// Health checks if the embedding service is healthy
	Health(ctx context.Context) error

	// Readiness checks if the embedding service is ready to serve requests
	Readiness(ctx context.Context) error

	// Close releases resources
	Close() error
}

// EmbeddingConfig holds configuration for embedding service
type EmbeddingConfig struct {
	ModelPath        string // Path to ONNX model file
	MaxWorkers       int    // Number of concurrent inference workers
	MaxTextLength    int    // Maximum text length to process
	InferenceTimeout int    // Timeout in milliseconds
	CacheSize        int    // LRU cache size for embeddings
}
