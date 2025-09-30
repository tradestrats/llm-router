package routing

import (
	"context"
	"fmt"

	"llm-router/domain/bandit"
	"llm-router/domain/embedding"
	"llm-router/domain/persistence"
	banditinf "llm-router/infrastructure/bandit"
	infraembedding "llm-router/infrastructure/embedding"
	"llm-router/infrastructure/openrouter"

	"github.com/sirupsen/logrus"
	"gorm.io/gorm"
)

// ContextualRoutingConfig holds configuration for contextual routing setup
type ContextualRoutingConfig struct {
	// Embedding configuration
	EmbeddingServiceType string
	EmbeddingServiceURL  string
	EmbeddingModelPath   string
	MaxEmbeddingWorkers  int
	EmbeddingCacheSize   int
	EmbeddingTimeout     int // milliseconds

	// Bandit configuration
	SimilarityThreshold float64
	MaxSimilarRequests  int
	RecencyDays         int
	FeedbackWeight      float64
	LatencyWeight       float64
	CostWeight          float64
	ExplorationRate     float64
	MinSimilarRequests  int
	DefaultModel        string

	// State management
	BatchSize int

	// Cold start and resilience configuration
	MinConfidenceScore   float64 // Minimum confidence for bandit decisions
	OptimisticPrior      float64 // Prior success rate for new models
	ExplorationBonus     float64 // Bonus score for exploration
	MinRequestsForGlobal int     // Minimum requests before trusting global arms
}

// DefaultContextualRoutingConfig returns sensible defaults
func DefaultContextualRoutingConfig() ContextualRoutingConfig {
	return ContextualRoutingConfig{
		EmbeddingServiceType: "http",
		EmbeddingServiceURL:  "http://localhost:8001",
		EmbeddingModelPath:   "./models/all-MiniLM-L6-v2",
		MaxEmbeddingWorkers:  4,
		EmbeddingCacheSize:   1000,
		EmbeddingTimeout:     5000, // 5 seconds

		SimilarityThreshold: 0.7,
		MaxSimilarRequests:  50,
		RecencyDays:         30,
		FeedbackWeight:      0.6,
		LatencyWeight:       0.2,  // Positive because normalizedLatency rewards speed
		CostWeight:          -0.2, // Negative because lower is better
		ExplorationRate:     0.1,
		MinSimilarRequests:  5,
		DefaultModel:        "anthropic/claude-3.5-sonnet",

		BatchSize: 100,
	}
}

// ContextualRoutingFactory creates and wires up contextual routing components
type ContextualRoutingFactory struct {
	config           ContextualRoutingConfig
	db               *gorm.DB
	requestRepo      persistence.RequestRepository
	embeddingService embedding.EmbeddingService
	stateManager     *banditinf.StateManager
	router           bandit.SimilarityRouter
}

// NewContextualRoutingFactory creates a new factory for contextual routing
func NewContextualRoutingFactory(config ContextualRoutingConfig, db *gorm.DB, requestRepo persistence.RequestRepository) (*ContextualRoutingFactory, error) {
	factory := &ContextualRoutingFactory{
		config:      config,
		db:          db,
		requestRepo: requestRepo,
	}

	if err := factory.initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize contextual routing factory: %w", err)
	}

	return factory, nil
}

// initialize sets up all the contextual routing components
func (f *ContextualRoutingFactory) initialize() error {
	var err error

	// 1. Initialize embedding service using factory
	embeddingConfig := embedding.EmbeddingConfig{
		ModelPath:        f.config.EmbeddingModelPath,
		MaxWorkers:       f.config.MaxEmbeddingWorkers,
		MaxTextLength:    8192, // Standard limit
		InferenceTimeout: f.config.EmbeddingTimeout,
		CacheSize:        f.config.EmbeddingCacheSize,
	}

	// Create embedding service factory
	serviceType := infraembedding.EmbeddingServiceType(f.config.EmbeddingServiceType)
	factory := infraembedding.NewEmbeddingServiceFactory(serviceType, embeddingConfig, f.config.EmbeddingServiceURL)

	f.embeddingService, err = factory.CreateEmbeddingService()
	if err != nil {
		return fmt.Errorf("failed to create embedding service: %w", err)
	}

	// 2. Initialize simplified bandit state manager
	f.stateManager, err = banditinf.NewStateManager(f.db, f.requestRepo)
	if err != nil {
		return fmt.Errorf("failed to create state manager: %w", err)
	}

	// 3. Initialize contextual router with enhanced cold start configuration
	banditConfig := bandit.BanditConfig{
		SimilarityThreshold:  f.config.SimilarityThreshold,
		MaxSimilarRequests:   f.config.MaxSimilarRequests,
		RecencyDays:          f.config.RecencyDays,
		FeedbackWeight:       f.config.FeedbackWeight,
		LatencyWeight:        f.config.LatencyWeight,
		CostWeight:           f.config.CostWeight,
		ExplorationRate:      f.config.ExplorationRate,
		MinSimilarRequests:   f.config.MinSimilarRequests,
		MinConfidenceScore:   f.config.MinConfidenceScore,
		OptimisticPrior:      f.config.OptimisticPrior,
		ExplorationBonus:     f.config.ExplorationBonus,
		MinRequestsForGlobal: f.config.MinRequestsForGlobal,
		GlobalFallback:       true,
		DefaultModel:         f.config.DefaultModel,
	}

	f.router, err = banditinf.NewContextualRouter(banditConfig, f.embeddingService, f.stateManager, f.db)
	if err != nil {
		return fmt.Errorf("failed to create contextual router: %w", err)
	}

	logrus.Info("Contextual routing components initialized successfully")
	return nil
}

// SetContextualRouterInterface defines the interface for providers that support contextual routing
type SetContextualRouterInterface interface {
	SetContextualRouter(router bandit.SimilarityRouter)
}

// AttachToProvider attaches the contextual router to an OpenRouter provider
func (f *ContextualRoutingFactory) AttachToProvider(provider *openrouter.Provider) {
	if f.router != nil {
		provider.SetContextualRouter(f.router)
		logrus.Info("Contextual router attached to provider")
	}
}

// AttachToProviderInterface attaches the contextual router to any provider that supports it
func (f *ContextualRoutingFactory) AttachToProviderInterface(provider SetContextualRouterInterface) {
	if f.router != nil {
		provider.SetContextualRouter(f.router)
		logrus.Info("Contextual router attached to provider interface")
	}
}

// GetEmbeddingService returns the embedding service (for embedding persistence)
func (f *ContextualRoutingFactory) GetEmbeddingService() embedding.EmbeddingService {
	return f.embeddingService
}

// GetStateManager returns the state manager (for performance updates)
func (f *ContextualRoutingFactory) GetStateManager() *banditinf.StateManager {
	return f.stateManager
}

// GetRouter returns the contextual router
func (f *ContextualRoutingFactory) GetRouter() bandit.SimilarityRouter {
	return f.router
}

// Health checks if all contextual routing components are healthy
func (f *ContextualRoutingFactory) Health(ctx context.Context) error {
	if f.embeddingService != nil {
		if err := f.embeddingService.Health(ctx); err != nil {
			return fmt.Errorf("embedding service unhealthy: %w", err)
		}
	}

	if f.stateManager != nil {
		if err := f.stateManager.Health(); err != nil {
			return fmt.Errorf("state manager unhealthy: %w", err)
		}
	}

	if f.router != nil {
		if err := f.router.Health(ctx); err != nil {
			return fmt.Errorf("contextual router unhealthy: %w", err)
		}
	}

	return nil
}

// Readiness checks if all contextual routing components are ready to serve requests
func (f *ContextualRoutingFactory) Readiness(ctx context.Context) error {
	if f.embeddingService != nil {
		if err := f.embeddingService.Readiness(ctx); err != nil {
			return fmt.Errorf("embedding service not ready: %w", err)
		}
	}

	if f.stateManager != nil {
		if err := f.stateManager.Health(); err != nil {
			return fmt.Errorf("state manager not ready: %w", err)
		}
	}

	if f.router != nil {
		if err := f.router.Health(ctx); err != nil {
			return fmt.Errorf("contextual router not ready: %w", err)
		}
	}

	return nil
}

// Close gracefully shuts down all contextual routing components
func (f *ContextualRoutingFactory) Close() error {
	var errors []error

	if f.embeddingService != nil {
		if err := f.embeddingService.Close(); err != nil {
			errors = append(errors, fmt.Errorf("failed to close embedding service: %w", err))
		}
	}

	if f.stateManager != nil {
		if err := f.stateManager.Close(); err != nil {
			errors = append(errors, fmt.Errorf("failed to close state manager: %w", err))
		}
	}

	if f.router != nil {
		if err := f.router.Close(); err != nil {
			errors = append(errors, fmt.Errorf("failed to close router: %w", err))
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("errors during contextual routing shutdown: %v", errors)
	}

	logrus.Info("Contextual routing components closed successfully")
	return nil
}
