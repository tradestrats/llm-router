package persistence

import (
	"context"

	"github.com/google/uuid"
)

// Repository defines the generic repository interface using Go generics
type Repository[T any] interface {
	Create(ctx context.Context, entity *T) error
	Update(ctx context.Context, entity *T) error
	FindByID(ctx context.Context, id uuid.UUID) (*T, error)
	Delete(ctx context.Context, id uuid.UUID) error
}

// RequestRepository defines operations specific to request records
type RequestRepository interface {
	Repository[RequestRecord]

	// Request-specific operations
	FindByIDWithRelations(ctx context.Context, id uuid.UUID) (*RequestRecord, error)
	FindByStatus(ctx context.Context, status RequestStatus, limit int) ([]*RequestRecord, error)
	FindRecent(ctx context.Context, limit int) ([]*RequestRecord, error)
	UpdateStatus(ctx context.Context, id uuid.UUID, status RequestStatus) error
}

// MetricsRepository defines operations for request metrics
type MetricsRepository interface {
	Repository[RequestMetrics]

	// Metrics-specific operations
	FindByRequestID(ctx context.Context, requestID uuid.UUID) (*RequestMetrics, error)
	CreateOrUpdate(ctx context.Context, metrics *RequestMetrics) error
	GetAggregatedMetrics(ctx context.Context, limit int) (*AggregatedMetrics, error)
}

// FeedbackRepository defines operations for request feedback
type FeedbackRepository interface {
	Repository[RequestFeedback]

	// Feedback-specific operations
	FindByRequestID(ctx context.Context, requestID uuid.UUID) ([]*RequestFeedback, error)
	GetAverageScore(ctx context.Context, requestID *uuid.UUID) (float64, error)
	FindRecentFeedback(ctx context.Context, limit int) ([]*RequestFeedback, error)
}

// EventProcessor defines the interface for processing persistence events asynchronously
type EventProcessor interface {
	// Start begins processing events from the channel
	Start(ctx context.Context) error

	// Stop gracefully shuts down the event processor
	Stop() error

	// ProcessEvent sends an event to be processed asynchronously
	ProcessEvent(event interface{}) error

	// Health returns the health status of the processor
	Health() ProcessorHealth
}

// ProcessorHealth represents the health status of the event processor
type ProcessorHealth struct {
	IsRunning      bool  `json:"is_running"`
	QueueSize      int   `json:"queue_size"`
	ProcessedCount int64 `json:"processed_count"`
	ErrorCount     int64 `json:"error_count"`
}

// AggregatedMetrics represents aggregated performance metrics
type AggregatedMetrics struct {
	TotalRequests    int64   `json:"total_requests"`
	AverageCost      float64 `json:"average_cost"`
	AverageTokens    float64 `json:"average_tokens"`
	AverageLatencyMs float64 `json:"average_latency_ms"`
	AverageFeedback  float64 `json:"average_feedback"`
	TotalCost        float64 `json:"total_cost"`
	TotalTokens      int64   `json:"total_tokens"`
}

// DatabaseManager defines the interface for database management operations
type DatabaseManager interface {
	// Connect establishes database connection
	Connect(ctx context.Context, dsn string) error

	// Close closes the database connection
	Close() error

	// Migrate runs database migrations
	Migrate() error

	// Health checks database connectivity
	Health(ctx context.Context) error

	// GetRepositories returns initialized repositories
	GetRepositories() (RequestRepository, MetricsRepository, FeedbackRepository)
}

// TransactionManager defines interface for database transactions
type TransactionManager interface {
	// WithTransaction executes a function within a database transaction
	WithTransaction(ctx context.Context, fn func(ctx context.Context) error) error
}

// RequestTracker defines the interface for tracking requests through their lifecycle
type RequestTracker interface {
	// StartTracking begins tracking a new request
	StartTracking(ctx context.Context, requestID uuid.UUID, requestData []byte, model string, isStreaming bool) error

	// CompleteTracking finalizes request tracking with response data
	CompleteTracking(ctx context.Context, requestID uuid.UUID, responseData []byte, metrics RequestMetrics) error

	// FailTracking marks a request as failed
	FailTracking(ctx context.Context, requestID uuid.UUID, errorMsg string) error

	// SubmitFeedback adds feedback for a request
	SubmitFeedback(ctx context.Context, requestID uuid.UUID, feedbackText string, score float64) error

	// UpdateEmbedding updates the embedding for a request
	UpdateEmbedding(ctx context.Context, requestID uuid.UUID, embedding []float32) error
}
