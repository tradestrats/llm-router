package bandit

import (
	"context"
	"llm-router/domain/chat"
)

// SimilarityRouter defines the interface for contextual bandit routing
type SimilarityRouter interface {
	// SelectModel selects the best model for the given request using Thompson sampling
	SelectModel(ctx context.Context, req *chat.Request) (string, error)

	// UpdatePerformance updates the bandit state with performance metrics
	UpdatePerformance(ctx context.Context, requestID string, model string, metrics PerformanceMetrics) error

	// Health checks if the router is healthy
	Health(ctx context.Context) error

	// Close releases resources
	Close() error
}

// StateManager defines the interface for managing bandit arm state
type StateManager interface {
	// GetArm returns a copy of the bandit arm for the specified model
	GetArm(model string) *BanditArm

	// GetAllArms returns copies of all bandit arms
	GetAllArms() map[string]*BanditArm

	// UpdateArm updates the bandit arm with new performance metrics
	UpdateArm(model string, metrics PerformanceMetrics) error

	// CreateArm creates a new bandit arm with optimistic initialization
	CreateArm(model string) error

	// GetStatistics returns aggregated statistics about bandit performance
	GetStatistics() map[string]interface{}

	// Health checks if the state manager is healthy
	Health() error

	// Close shuts down the state manager gracefully
	Close() error
}

// PerformanceMetrics holds performance data for bandit updates
type PerformanceMetrics struct {
	Latency      float64 // Latency in milliseconds
	Cost         float64 // Cost in dollars
	FeedbackScore float64 // User feedback score (0-1)
	Success      bool    // Whether the request succeeded
}

// SimilarRequest represents a similar request found via embedding search
type SimilarRequest struct {
	RequestID  string  `json:"request_id"`
	Model      string  `json:"model"`
	Similarity float64 `json:"similarity"`
	Latency    float64 `json:"latency"`
	Cost       float64 `json:"cost"`
	Feedback   float64 `json:"feedback"`
	TokensUsed int     `json:"tokens_used"`
	CreatedAt  string  `json:"created_at"`
}

// BanditConfig holds configuration for the bandit router
type BanditConfig struct {
	// Similarity search parameters
	SimilarityThreshold float64 // Minimum cosine similarity (0-1)
	MaxSimilarRequests  int     // Maximum similar requests to consider (top-K)
	RecencyDays        int     // Only consider requests from last N days

	// Thompson sampling parameters
	FeedbackWeight    float64 // Weight for feedback score in reward calculation
	LatencyWeight     float64 // Weight for latency in reward calculation (positive, rewards speed)
	CostWeight        float64 // Weight for cost in reward calculation (negative)
	ExplorationRate   float64 // Exploration vs exploitation balance (0-1)

	// Cold start and resilience parameters
	MinSimilarRequests   int     // Minimum similar requests needed to use similarity routing
	MinConfidenceScore   float64 // Minimum confidence needed to trust bandit decision
	OptimisticPrior      float64 // Prior success rate for new models (0-1)
	ExplorationBonus     float64 // Bonus score for models with little data
	MinRequestsForGlobal int     // Minimum requests needed before trusting global arms

	// Fallback parameters
	GlobalFallback bool   // Whether to fallback to global Thompson sampling
	DefaultModel   string // Ultimate fallback model
}

// BanditArm represents a model's performance statistics
type BanditArm struct {
	Model           string
	RequestCount    int
	SuccessCount    int
	TotalLatency    float64
	TotalCost       float64
	TotalFeedback   float64
	FeedbackCount   int
	LastUpdated     string
}