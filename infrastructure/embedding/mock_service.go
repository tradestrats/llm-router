package embedding

import (
	"context"
	"crypto/sha256"
	"fmt"
	"strings"
	"sync"
	"time"

	"llm-router/domain/chat"
	"llm-router/domain/embedding"

	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/sirupsen/logrus"
)

// MockEmbeddingService implements embedding.EmbeddingService using mock embeddings
type MockEmbeddingService struct {
	config      embedding.EmbeddingConfig
	workers     chan struct{} // Semaphore for worker pool
	cache       *lru.Cache[string, []float32]
	mu          sync.RWMutex
	initialized bool
	closed      bool
}

// NewMockEmbeddingService creates a new mock embedding service
func NewMockEmbeddingService(config embedding.EmbeddingConfig) (*MockEmbeddingService, error) {
	// Validate config
	if config.MaxWorkers <= 0 {
		config.MaxWorkers = 4
	}
	if config.MaxTextLength <= 0 {
		config.MaxTextLength = 8192
	}
	if config.InferenceTimeout <= 0 {
		config.InferenceTimeout = 5000 // 5 seconds
	}
	if config.CacheSize <= 0 {
		config.CacheSize = 1000
	}

	// Create LRU cache for embeddings
	cache, err := lru.New[string, []float32](config.CacheSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create cache: %w", err)
	}

	service := &MockEmbeddingService{
		config:  config,
		workers: make(chan struct{}, config.MaxWorkers),
		cache:   cache,
	}

	// Initialize the service
	if err := service.initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize mock embedding service: %w", err)
	}

	return service, nil
}

// initialize sets up the mock embedding service
func (s *MockEmbeddingService) initialize() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.initialized {
		return nil
	}

	logrus.Info("Initializing mock embedding service")

	s.initialized = true
	logrus.Info("Mock embedding service initialized successfully")
	return nil
}

// Embed generates mock embeddings for the given text
func (s *MockEmbeddingService) Embed(ctx context.Context, text string) ([]float32, error) {
	if err := s.checkHealth(); err != nil {
		return nil, err
	}

	// Validate input
	if len(text) > s.config.MaxTextLength {
		text = text[:s.config.MaxTextLength]
		logrus.WithField("original_length", len(text)).Warn("Text truncated to maximum length")
	}

	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("empty text provided")
	}

	// Check cache first
	cacheKey := s.getCacheKey(text)
	if cached, ok := s.cache.Get(cacheKey); ok {
		return cached, nil
	}

	// Acquire worker semaphore
	select {
	case s.workers <- struct{}{}:
		defer func() { <-s.workers }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	// Create context with timeout
	inferenceCtx, cancel := context.WithTimeout(ctx, time.Duration(s.config.InferenceTimeout)*time.Millisecond)
	defer cancel()

	// Generate mock embedding
	embedding, err := s.generateMockEmbedding(inferenceCtx, text)
	if err != nil {
		return nil, fmt.Errorf("failed to generate mock embedding: %w", err)
	}

	// Cache the result
	s.cache.Add(cacheKey, embedding)

	return embedding, nil
}

// EmbedMessages generates mock embeddings for chat messages
func (s *MockEmbeddingService) EmbedMessages(ctx context.Context, messages []chat.Message) ([]float32, error) {
	if len(messages) == 0 {
		return nil, fmt.Errorf("no messages provided")
	}

	// Concatenate messages with role prefixes
	var textParts []string
	for _, msg := range messages {
		textParts = append(textParts, fmt.Sprintf("%s: %s", msg.Role, msg.Content))
	}

	text := strings.Join(textParts, "\n")
	return s.Embed(ctx, text)
}

// BatchEmbed generates mock embeddings for multiple texts with parallel processing
func (s *MockEmbeddingService) BatchEmbed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("no texts provided")
	}

	// Limit batch size to prevent resource exhaustion
	maxBatchSize := 100
	if len(texts) > maxBatchSize {
		return nil, fmt.Errorf("batch size %d exceeds maximum %d", len(texts), maxBatchSize)
	}

	// Use goroutines for parallel processing with worker pool
	type result struct {
		index     int
		embedding []float32
		err       error
	}

	results := make([][]float32, len(texts))
	resultChan := make(chan result, len(texts))

	// Process texts in parallel
	for i, text := range texts {
		go func(index int, text string) {
			// Acquire worker semaphore
			select {
			case s.workers <- struct{}{}:
				defer func() { <-s.workers }()
			case <-ctx.Done():
				resultChan <- result{index: index, err: ctx.Err()}
				return
			}

			embedding, err := s.Embed(ctx, text)
			resultChan <- result{index: index, embedding: embedding, err: err}
		}(i, text)
	}

	// Collect results
	for i := 0; i < len(texts); i++ {
		select {
		case res := <-resultChan:
			if res.err != nil {
				return nil, fmt.Errorf("failed to embed text at index %d: %w", res.index, res.err)
			}
			results[res.index] = res.embedding
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	return results, nil
}

// generateMockEmbedding performs mock embedding generation
func (s *MockEmbeddingService) generateMockEmbedding(ctx context.Context, text string) ([]float32, error) {
	// Simulate some processing time
	select {
	case <-time.After(10 * time.Millisecond):
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	// Generate deterministic mock embedding (384 dimensions filled with hash-based values)
	hash := sha256.Sum256([]byte(text))
	embedding := make([]float32, 384)
	for i := 0; i < 384; i++ {
		// Generate deterministic float values from hash
		embedding[i] = float32(hash[i%32]) / 255.0
	}

	logrus.WithField("text_length", len(text)).Debug("Generated mock embedding")
	return embedding, nil
}

// GetDimensions returns the dimensionality of embeddings
func (s *MockEmbeddingService) GetDimensions() int {
	return 384 // all-MiniLM-L6-v2 dimensions
}

// Health checks if the embedding service is healthy
func (s *MockEmbeddingService) Health(ctx context.Context) error {
	return s.checkHealth()
}

// Readiness checks if the embedding service is ready to serve requests
func (s *MockEmbeddingService) Readiness(ctx context.Context) error {
	return s.checkReadiness()
}

// checkHealth performs internal health checks
func (s *MockEmbeddingService) checkHealth() error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return fmt.Errorf("mock embedding service is closed")
	}

	if !s.initialized {
		return fmt.Errorf("mock embedding service not initialized")
	}

	return nil
}

// checkReadiness performs readiness checks for mock service
func (s *MockEmbeddingService) checkReadiness() error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return fmt.Errorf("mock embedding service is closed")
	}

	if !s.initialized {
		return fmt.Errorf("mock embedding service not initialized")
	}

	// Mock service is always ready when initialized
	return nil
}

// Close releases resources
func (s *MockEmbeddingService) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil
	}

	s.closed = true

	logrus.Info("Mock embedding service closed")
	return nil
}

// getCacheKey generates a cache key for the given text
func (s *MockEmbeddingService) getCacheKey(text string) string {
	hash := sha256.Sum256([]byte(text))
	return fmt.Sprintf("mock_%x", hash[:16]) // Use first 16 bytes for shorter key
}
