package chat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"llm-router/domain/bandit"
	"llm-router/domain/chat"
	"llm-router/domain/embedding"
	"llm-router/domain/persistence"
	"strings"
	"time"

	"github.com/google/uuid"
)

// Service orchestrates chat use cases
type Service struct {
	provider         chat.ProviderPort
	stream           chat.StreamProviderPort[chat.StreamChunk]
	tracker          persistence.RequestTracker
	embeddingService embedding.EmbeddingService
	banditRouter     bandit.SimilarityRouter // For performance feedback
}

// RequestContext holds request-specific information
type RequestContext struct {
	RequestID uuid.UUID
	StartTime time.Time
	Model     string
}

func NewService(provider chat.ProviderPort, stream chat.StreamProviderPort[chat.StreamChunk], tracker persistence.RequestTracker, embeddingService embedding.EmbeddingService, banditRouter bandit.SimilarityRouter) *Service {
	return &Service{
		provider:         provider,
		stream:           stream,
		tracker:          tracker,
		embeddingService: embeddingService,
		banditRouter:     banditRouter,
	}
}

// NewServiceWithoutTracking creates a service without request tracking (for backward compatibility)
func NewServiceWithoutTracking(provider chat.ProviderPort, stream chat.StreamProviderPort[chat.StreamChunk]) *Service {
	return &Service{
		provider:         provider,
		stream:           stream,
		tracker:          nil, // No tracking
		embeddingService: nil, // No embedding service
		banditRouter:     nil, // No bandit feedback
	}
}

func (s *Service) Chat(ctx context.Context, req *chat.Request) (*chat.Response, error) {
	if len(req.Messages) == 0 {
		return nil, errors.New("messages cannot be empty")
	}
	if req.Stream {
		return nil, errors.New("use Stream for streaming requests")
	}

	// Validate request size
	const maxMessages = 100
	if len(req.Messages) > maxMessages {
		return nil, fmt.Errorf("too many messages: %d (max %d)", len(req.Messages), maxMessages)
	}

	// Validate message content
	for i, msg := range req.Messages {
		if msg.Role == "" {
			return nil, fmt.Errorf("message %d: role cannot be empty", i)
		}
		if msg.Content == "" {
			return nil, fmt.Errorf("message %d: content cannot be empty", i)
		}
		const maxContentLength = 50000
		if len(msg.Content) > maxContentLength {
			return nil, fmt.Errorf("message %d: content too long (%d chars, max %d)", i, len(msg.Content), maxContentLength)
		}
		if msg.Role != "user" && msg.Role != "assistant" && msg.Role != "system" {
			return nil, fmt.Errorf("message %d: invalid role '%s' (must be user, assistant, or system)", i, msg.Role)
		}
	}

	// Create request context for tracking
	requestID := uuid.New()
	// Try to get UUID from context if available (from middleware)
	if uuidStr, ok := ctx.Value("request_uuid").(string); ok {
		if parsedUUID, err := uuid.Parse(uuidStr); err == nil {
			requestID = parsedUUID
		}
	}
	startTime := time.Now()

	// Execute the actual chat request first to get the model
	resp, err := s.provider.Chat(ctx, req)

	// Calculate metrics
	latency := time.Since(startTime)

	// Determine model for tracking (use response model if available, fallback to "unknown")
	model := "unknown"
	if resp != nil && resp.Model != "" {
		model = resp.Model
	}

	if err != nil {
		// Start tracking for failed request if tracker is available
		if s.tracker != nil {
			go func(parentCtx context.Context) {
				// Serialize request data for failed tracking
				requestData, serErr := json.Marshal(req)
				if serErr != nil {
					fmt.Printf("Failed to serialize request for failed tracking: %v\n", serErr)
					return
				}

				// Start tracking then immediately fail it - use original context to preserve values
				opCtx, cancel := context.WithTimeout(parentCtx, 5*time.Second)
				if trackErr := s.tracker.StartTracking(opCtx, requestID, requestData, model, false); trackErr != nil {
					fmt.Printf("Failed to start tracking failed request %s: %v\n", requestID, trackErr)
					cancel()
					return
				}
				if trackErr := s.tracker.FailTracking(opCtx, requestID, err.Error()); trackErr != nil {
					fmt.Printf("Failed to track request failure %s: %v\n", requestID, trackErr)
				}

				// Update bandit with failure
				if s.banditRouter != nil {
					banditMetrics := bandit.PerformanceMetrics{
						Latency:       float64(latency.Milliseconds()),
						Cost:          0, // No cost on failure
						FeedbackScore: 0, // No feedback on failure
						Success:       false,
					}

					if banditErr := s.banditRouter.UpdatePerformance(opCtx, requestID.String(), model, banditMetrics); banditErr != nil {
						fmt.Printf("Failed to update bandit with failure for request %s: %v\n", requestID, banditErr)
					}
				}

				cancel()
			}(ctx)
		}
		return nil, err
	}

	// Start and complete tracking with success metrics
	if s.tracker != nil {
		go func(parentCtx context.Context) {
			// Serialize request data
			requestData, serErr := json.Marshal(req)
			if serErr != nil {
				fmt.Printf("Failed to serialize request for tracking: %v\n", serErr)
				return
			}

			// Start tracking with actual model - use the original context to preserve values
			opCtx, cancel := context.WithTimeout(parentCtx, 5*time.Second)
			if trackErr := s.tracker.StartTracking(opCtx, requestID, requestData, model, false); trackErr != nil {
				fmt.Printf("Failed to start tracking request %s: %v\n", requestID, trackErr)
				cancel()
				return
			}

			responseData, _ := json.Marshal(resp)

			// Calculate cost using actual model
			cost := s.calculateCost(resp.Model, resp.Usage)

			metrics := persistence.RequestMetrics{
				TotalCost:  cost,
				TokensUsed: resp.Usage.TotalTokens,
				LatencyMs:  latency.Milliseconds(),
			}

			if trackErr := s.tracker.CompleteTracking(opCtx, requestID, responseData, metrics); trackErr != nil {
				fmt.Printf("Failed to complete tracking request %s: %v\n", requestID, trackErr)
			}

			// Update bandit performance asynchronously
			if s.banditRouter != nil {
				go func() {
					banditCtx, banditCancel := context.WithTimeout(context.WithoutCancel(parentCtx), 3*time.Second)
					defer banditCancel()

					banditMetrics := bandit.PerformanceMetrics{
						Latency:       float64(latency.Milliseconds()),
						Cost:          cost,
						FeedbackScore: 0, // Will be updated when user provides feedback
						Success:       true,
					}

					if err := s.banditRouter.UpdatePerformance(banditCtx, requestID.String(), model, banditMetrics); err != nil {
						fmt.Printf("Failed to update bandit performance for request %s: %v\n", requestID, err)
					}
				}()
			}

			// Generate embedding asynchronously (non-blocking)
			go func() {
				embedCtx, embedCancel := context.WithTimeout(context.WithoutCancel(parentCtx), 10*time.Second)
				defer embedCancel()

				// Try to generate embedding for the request
				if embedding, err := s.generateEmbedding(embedCtx, req.Messages); err == nil && len(embedding) > 0 {
					if updateErr := s.tracker.UpdateEmbedding(embedCtx, requestID, embedding); updateErr != nil {
						fmt.Printf("Failed to update embedding for request %s: %v\n", requestID, updateErr)
					}
				} else if err != nil {
					fmt.Printf("Failed to generate embedding for request %s: %v\n", requestID, err)
				}
			}()

			cancel()
		}(ctx)
	}

	return resp, nil
}

// calculateCost extracts cost from the LLM provider response or falls back to 0.0
func (s *Service) calculateCost(model string, usage chat.Usage) float64 {
	// Use cost from OpenRouter response if available
	if usage.Cost != nil {
		return *usage.Cost
	}

	// If no cost provided by provider, return 0.0
	// The usage metrics are still tracked accurately for external cost calculation
	return 0.0
}

// generateEmbedding generates an embedding for the given messages with consistent field ordering
func (s *Service) generateEmbedding(ctx context.Context, messages []chat.Message) ([]float32, error) {
	// Return early if no embedding service is available
	if s.embeddingService == nil {
		return nil, fmt.Errorf("embedding service not available")
	}

	// Prepare text with consistent field ordering and smart truncation
	text := s.prepareTextForEmbedding(messages)

	// Generate embedding using the embedding service
	embedding, err := s.embeddingService.Embed(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	return embedding, nil
}

// prepareTextForEmbedding creates consistently ordered text from messages with smart truncation
func (s *Service) prepareTextForEmbedding(messages []chat.Message) string {
	const maxEmbeddingLength = 8000 // Conservative limit for embedding models
	const truncationMarker = "[TRUNCATED]"

	var parts []string
	var systemParts []string
	var userParts []string
	var assistantParts []string

	// Group messages by role for consistent ordering
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			systemParts = append(systemParts, fmt.Sprintf("[SYSTEM] %s", msg.Content))
		case "user":
			userParts = append(userParts, fmt.Sprintf("[USER] %s", msg.Content))
		case "assistant":
			assistantParts = append(assistantParts, fmt.Sprintf("[ASSISTANT] %s", msg.Content))
		default:
			// Handle unknown roles consistently
			userParts = append(userParts, fmt.Sprintf("[%s] %s", strings.ToUpper(msg.Role), msg.Content))
		}
	}

	// Build text in consistent field order: system, user, assistant
	parts = append(parts, systemParts...)
	parts = append(parts, userParts...)
	parts = append(parts, assistantParts...)

	// Join with consistent separators
	fullText := strings.Join(parts, "\n")

	// Apply smart truncation if needed
	if len(fullText) > maxEmbeddingLength {
		// Always preserve metadata and structure markers
		metadataPrefixes := []string{"[SYSTEM]", "[USER]", "[ASSISTANT]", "[TRUNCATED]"}

		// Find last part that fits within limit, keeping metadata intact
		truncatedText := s.smartTruncate(fullText, maxEmbeddingLength, metadataPrefixes)
		return fmt.Sprintf("%s\n%s", truncationMarker, truncatedText)
	}

	return fullText
}

// smartTruncate truncates text while preserving important metadata fields and keeping the last part
func (s *Service) smartTruncate(text string, maxLength int, preservePrefixes []string) string {
	if len(text) <= maxLength {
		return text
	}

	// Calculate space needed for truncation marker and buffer
	const bufferSize = 100
	availableLength := maxLength - bufferSize

	// If text is much longer, take the last portion to preserve recent context
	if len(text) > availableLength*2 {
		// Keep the last portion of the text
		startPos := len(text) - availableLength
		truncated := text[startPos:]

		// Try to start at a clean boundary (after a newline or field marker)
		for i, char := range truncated {
			if char == '\n' {
				return truncated[i+1:]
			}
			// Also check for field markers
			for _, prefix := range preservePrefixes {
				if strings.HasPrefix(truncated[i:], prefix) {
					return truncated[i:]
				}
			}
		}

		return truncated
	}

	// For shorter text, just truncate from the end
	return text[:availableLength]
}

func (s *Service) Stream(ctx context.Context, req *chat.Request, onChunk chat.StreamHandler[chat.StreamChunk]) error {
	if len(req.Messages) == 0 {
		return errors.New("messages cannot be empty")
	}
	if !req.Stream {
		return errors.New("set stream=true for streaming")
	}

	// Apply same validation as Chat method
	const maxMessages = 100
	if len(req.Messages) > maxMessages {
		return fmt.Errorf("too many messages: %d (max %d)", len(req.Messages), maxMessages)
	}

	for i, msg := range req.Messages {
		if msg.Role == "" {
			return fmt.Errorf("message %d: role cannot be empty", i)
		}
		if msg.Content == "" {
			return fmt.Errorf("message %d: content cannot be empty", i)
		}
		const maxContentLength = 50000
		if len(msg.Content) > maxContentLength {
			return fmt.Errorf("message %d: content too long (%d chars, max %d)", i, len(msg.Content), maxContentLength)
		}
		if msg.Role != "user" && msg.Role != "assistant" && msg.Role != "system" {
			return fmt.Errorf("message %d: invalid role '%s' (must be user, assistant, or system)", i, msg.Role)
		}
	}

	// Create request context for tracking
	requestID := uuid.New()
	// Try to get UUID from context if available (from middleware)
	if uuidStr, ok := ctx.Value("request_uuid").(string); ok {
		if parsedUUID, err := uuid.Parse(uuidStr); err == nil {
			requestID = parsedUUID
		}
	}
	startTime := time.Now()
	var finalUsage *chat.Usage
	var responseChunks []chat.StreamChunk
	var model string = "unknown"

	// Wrap the onChunk handler to capture usage, response data, and model
	wrappedHandler := func(chunk chat.StreamChunk) error {
		// Capture model from first chunk (all chunks should have same model)
		if chunk.Model != "" && model == "unknown" {
			model = chunk.Model

			// Start tracking now that we have the model
			if s.tracker != nil {
				go func() {
					requestData, serErr := json.Marshal(req)
					if serErr != nil {
						fmt.Printf("Failed to serialize request for stream tracking: %v\n", serErr)
						return
					}

					if err := s.tracker.StartTracking(context.Background(), requestID, requestData, model, true); err != nil {
						fmt.Printf("Failed to start tracking stream request %s: %v\n", requestID, err)
					}
				}()
			}
		}

		// Capture final usage from the last chunk
		if chunk.Usage != nil {
			finalUsage = chunk.Usage
		}

		// Store chunks for response reconstruction
		if s.tracker != nil {
			responseChunks = append(responseChunks, chunk)
		}

		return onChunk(chunk)
	}

	// Execute the streaming request
	err := s.stream.Stream(ctx, req, wrappedHandler)

	// Calculate metrics
	latency := time.Since(startTime)

	if err != nil {
		// Track failure if tracker is available and tracking was started
		if s.tracker != nil {
			go func() {
				// If tracking was never started (no model received), start it first
				if model == "unknown" {
					requestData, serErr := json.Marshal(req)
					if serErr != nil {
						fmt.Printf("Failed to serialize request for failed stream tracking: %v\n", serErr)
						return
					}

					if trackErr := s.tracker.StartTracking(context.Background(), requestID, requestData, model, true); trackErr != nil {
						fmt.Printf("Failed to start tracking failed stream request %s: %v\n", requestID, trackErr)
						return
					}
				}

				if trackErr := s.tracker.FailTracking(context.Background(), requestID, err.Error()); trackErr != nil {
					fmt.Printf("Failed to track stream failure %s: %v\n", requestID, trackErr)
				}
			}()
		}
		return err
	}

	// Complete tracking with success metrics
	if s.tracker != nil {
		go func() {
			// Reconstruct response from chunks
			responseData, _ := json.Marshal(map[string]interface{}{
				"streaming":   true,
				"chunks":      responseChunks,
				"final_usage": finalUsage,
			})

			var cost float64
			var tokens int
			if finalUsage != nil {
				// Use actual model for cost calculation
				cost = s.calculateCost(model, *finalUsage)
				tokens = finalUsage.TotalTokens
			}

			metrics := persistence.RequestMetrics{
				TotalCost:  cost,
				TokensUsed: tokens,
				LatencyMs:  latency.Milliseconds(),
			}

			if trackErr := s.tracker.CompleteTracking(context.Background(), requestID, responseData, metrics); trackErr != nil {
				fmt.Printf("Failed to complete tracking stream request %s: %v\n", requestID, trackErr)
			}
		}()
	}

	return nil
}
