package embedding

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"strings"
	"sync"
	"time"

	"llm-router/domain/chat"
	"llm-router/domain/embedding"

	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/sirupsen/logrus"
	"github.com/sony/gobreaker"
)

// HTTPEmbeddingService implements embedding.EmbeddingService using HTTP calls to Python service
type HTTPEmbeddingService struct {
	config        embedding.EmbeddingConfig
	serviceURL    string
	httpClient    *http.Client
	workers       chan struct{} // Semaphore for worker pool
	cache         *lru.Cache[string, []float32]
	mu            sync.RWMutex
	initialized   bool
	closed        bool
	circuitBreaker *gobreaker.CircuitBreaker
	rng           *rand.Rand
	rngMutex      sync.Mutex
}

// EmbedRequest represents the request payload for the Python embedding service
type EmbedRequest struct {
	Text string `json:"text"`
}

// EmbedResponse represents the response from the Python embedding service
type EmbedResponse struct {
	Embedding  []float32 `json:"embedding"`
	Dimensions int       `json:"dimensions"`
	ModelName  string    `json:"model_name"`
}

// BatchEmbedRequest represents the batch request payload
type BatchEmbedRequest struct {
	Texts []string `json:"texts"`
}

// BatchEmbedResponse represents the batch response
type BatchEmbedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
	Dimensions int         `json:"dimensions"`
	ModelName  string      `json:"model_name"`
	Count      int         `json:"count"`
}

// NewHTTPEmbeddingService creates a new HTTP-based embedding service
func NewHTTPEmbeddingService(config embedding.EmbeddingConfig, serviceURL string) (*HTTPEmbeddingService, error) {
	// Validate config
	if serviceURL == "" {
		return nil, fmt.Errorf("embedding service URL is required")
	}
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

	// Create HTTP client with connection pooling
	transport := &http.Transport{
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   100,
		MaxConnsPerHost:       100,
		IdleConnTimeout:       90 * time.Second,
		DisableCompression:    false,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		ResponseHeaderTimeout: 30 * time.Second,
	}

	// Create LRU cache for embeddings
	cache, err := lru.New[string, []float32](config.CacheSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create cache: %w", err)
	}

	// Configure circuit breaker for resilience
	cbSettings := gobreaker.Settings{
		Name:        "embedding-service",
		MaxRequests: 3,                     // Allow 3 requests in half-open state
		Interval:    60 * time.Second,      // Reset counts every minute
		Timeout:     30 * time.Second,      // Stay open for 30 seconds
		ReadyToTrip: func(counts gobreaker.Counts) bool {
			// Open circuit after 3 consecutive failures
			return counts.ConsecutiveFailures >= 3
		},
		OnStateChange: func(name string, from gobreaker.State, to gobreaker.State) {
			logrus.WithFields(logrus.Fields{
				"service": name,
				"from":    from,
				"to":      to,
			}).Warn("Embedding service circuit breaker state changed")
		},
	}

	service := &HTTPEmbeddingService{
		config:     config,
		serviceURL: serviceURL,
		httpClient: &http.Client{
			Timeout:   time.Duration(config.InferenceTimeout) * time.Millisecond,
			Transport: transport,
		},
		workers:        make(chan struct{}, config.MaxWorkers),
		cache:          cache,
		circuitBreaker: gobreaker.NewCircuitBreaker(cbSettings),
		rng:            rand.New(rand.NewSource(time.Now().UnixNano())),
	}

	// Initialize the service
	if err := service.initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize HTTP embedding service: %w", err)
	}

	return service, nil
}

// initialize sets up the HTTP embedding service
func (s *HTTPEmbeddingService) initialize() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.initialized {
		return nil
	}

	logrus.WithField("service_url", s.serviceURL).Info("Initializing HTTP embedding service")

	// Test connection to the Python service
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", s.serviceURL+"/health", nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to connect to embedding service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("embedding service health check failed with status: %d", resp.StatusCode)
	}

	s.initialized = true
	logrus.Info("HTTP embedding service initialized successfully")
	return nil
}

// Embed generates embeddings for the given text
func (s *HTTPEmbeddingService) Embed(ctx context.Context, text string) ([]float32, error) {
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

	// Generate embedding
	embedding, err := s.generateEmbedding(inferenceCtx, text)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Cache the result
	s.cache.Add(cacheKey, embedding)

	return embedding, nil
}

// EmbedMessages generates embeddings for chat messages
func (s *HTTPEmbeddingService) EmbedMessages(ctx context.Context, messages []chat.Message) ([]float32, error) {
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

// BatchEmbed generates embeddings for multiple texts with parallel processing
func (s *HTTPEmbeddingService) BatchEmbed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("no texts provided")
	}

	// Limit batch size to prevent overwhelming the service
	maxBatchSize := 50
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

// generateEmbedding performs the actual embedding generation using HTTP calls with circuit breaker and retry logic
func (s *HTTPEmbeddingService) generateEmbedding(ctx context.Context, text string) ([]float32, error) {
	maxRetries := 3
	var lastErr error

	for attempt := 0; attempt < maxRetries; attempt++ {
		if attempt > 0 {
			// Exponential backoff with jitter: 100ms, 200ms, 400ms + jitter
			s.rngMutex.Lock()
			base := time.Duration(math.Pow(2, float64(attempt-1))) * 100 * time.Millisecond
			jitter := time.Duration(s.rng.Intn(50)) * time.Millisecond
			s.rngMutex.Unlock()

			backoff := base + jitter
			logrus.WithFields(logrus.Fields{
				"attempt": attempt + 1,
				"backoff": backoff,
			}).Debug("Retrying embedding generation after backoff")

			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		// Execute through circuit breaker
		result, err := s.circuitBreaker.Execute(func() (interface{}, error) {
			return s.doEmbeddingRequest(ctx, text)
		})

		if err != nil {
			lastErr = err

			// If circuit breaker is open, fail fast without retry
			if err == gobreaker.ErrOpenState {
				logrus.Warn("Embedding service circuit breaker is open, failing fast")
				return nil, fmt.Errorf("embedding service circuit breaker open: %w", err)
			}

			logrus.WithFields(logrus.Fields{
				"attempt": attempt + 1,
				"error":   err,
			}).Debug("Embedding generation attempt failed")
			continue
		}

		// Success
		return result.([]float32), nil
	}

	return nil, fmt.Errorf("embedding generation failed after %d attempts: %w", maxRetries, lastErr)
}

// doEmbeddingRequest performs the actual HTTP request
func (s *HTTPEmbeddingService) doEmbeddingRequest(ctx context.Context, text string) ([]float32, error) {
	// Prepare request
	reqBody := EmbedRequest{Text: text}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", s.serviceURL+"/embed", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Make HTTP call
	resp, err := s.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embedding service returned status %d", resp.StatusCode)
	}

	// Parse response
	var embedResp EmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&embedResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return embedResp.Embedding, nil
}

// GetDimensions returns the dimensionality of embeddings
func (s *HTTPEmbeddingService) GetDimensions() int {
	return 384 // all-MiniLM-L6-v2 dimensions
}

// Health checks if the embedding service is healthy
func (s *HTTPEmbeddingService) Health(ctx context.Context) error {
	return s.checkHealth()
}

// Readiness checks if the embedding service is ready to serve requests
func (s *HTTPEmbeddingService) Readiness(ctx context.Context) error {
	return s.checkReadiness()
}

// checkHealth performs internal health checks with circuit breaker protection
func (s *HTTPEmbeddingService) checkHealth() error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return fmt.Errorf("HTTP embedding service is closed")
	}

	if !s.initialized {
		return fmt.Errorf("HTTP embedding service not initialized")
	}

	// Check if the Python service is healthy through circuit breaker
	_, err := s.circuitBreaker.Execute(func() (interface{}, error) {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		req, err := http.NewRequestWithContext(ctx, "GET", s.serviceURL+"/health", nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create health check request: %w", err)
		}

		resp, err := s.httpClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("failed to check embedding service health: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("embedding service health check failed with status: %d", resp.StatusCode)
		}

		return nil, nil
	})

	if err != nil {
		if err == gobreaker.ErrOpenState {
			return fmt.Errorf("embedding service circuit breaker is open")
		}
		return err
	}

	return nil
}

// checkReadiness performs readiness checks by calling the /ready endpoint with circuit breaker protection
func (s *HTTPEmbeddingService) checkReadiness() error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return fmt.Errorf("HTTP embedding service is closed")
	}

	if !s.initialized {
		return fmt.Errorf("HTTP embedding service not initialized")
	}

	// Check if the Python service is ready through circuit breaker
	_, err := s.circuitBreaker.Execute(func() (interface{}, error) {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		req, err := http.NewRequestWithContext(ctx, "GET", s.serviceURL+"/ready", nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create readiness check request: %w", err)
		}

		resp, err := s.httpClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("failed to check embedding service readiness: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("embedding service readiness check failed with status: %d", resp.StatusCode)
		}

		return nil, nil
	})

	if err != nil {
		if err == gobreaker.ErrOpenState {
			return fmt.Errorf("embedding service circuit breaker is open")
		}
		return err
	}

	return nil
}

// Close releases resources
func (s *HTTPEmbeddingService) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil
	}

	s.closed = true

	// Close HTTP client
	s.httpClient.CloseIdleConnections()

	logrus.Info("HTTP embedding service closed")
	return nil
}

// getCacheKey generates a cache key for the given text
func (s *HTTPEmbeddingService) getCacheKey(text string) string {
	hash := sha256.Sum256([]byte(text))
	return hex.EncodeToString(hash[:16]) // Use first 16 bytes for shorter key
}
