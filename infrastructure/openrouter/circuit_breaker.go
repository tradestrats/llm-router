package openrouter

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"llm-router/domain/bandit"
	"llm-router/domain/chat"

	"github.com/sirupsen/logrus"
	"github.com/sony/gobreaker"
)

// CircuitBreakerConfig holds configuration for circuit breaker behavior
type CircuitBreakerConfig struct {
	Enabled          bool          `yaml:"enabled" json:"enabled"`
	FailureThreshold uint32        `yaml:"failure_threshold" json:"failure_threshold"`
	SuccessThreshold uint32        `yaml:"success_threshold" json:"success_threshold"`
	Timeout          time.Duration `yaml:"timeout" json:"timeout"`
	MaxRequests      uint32        `yaml:"max_requests" json:"max_requests"`
}

// DefaultCircuitBreakerConfig returns sensible defaults for circuit breaker configuration
func DefaultCircuitBreakerConfig() CircuitBreakerConfig {
	return CircuitBreakerConfig{
		Enabled:          true,
		FailureThreshold: 5,                // Open after 5 consecutive failures
		SuccessThreshold: 2,                // Close after 2 successes in half-open state
		Timeout:          60 * time.Second, // Stay open for 60 seconds
		MaxRequests:      3,                // Allow max 3 requests in half-open state
	}
}

// CircuitBreakerProvider wraps a chat provider with circuit breaker functionality
// It maintains separate circuit breakers per model for granular failure isolation
type CircuitBreakerProvider struct {
	provider chat.ProviderPort
	stream   chat.StreamProviderPort[chat.StreamChunk]
	config   CircuitBreakerConfig
	breakers map[string]*gobreaker.CircuitBreaker
	mutex    sync.RWMutex
}

// NewCircuitBreakerProvider creates a new circuit breaker wrapper around a provider
func NewCircuitBreakerProvider(provider chat.ProviderPort, stream chat.StreamProviderPort[chat.StreamChunk], config CircuitBreakerConfig) *CircuitBreakerProvider {
	if !config.Enabled {
		// If circuit breaker is disabled, create a pass-through wrapper
		return &CircuitBreakerProvider{
			provider: provider,
			stream:   stream,
			config:   config,
			breakers: make(map[string]*gobreaker.CircuitBreaker),
		}
	}

	return &CircuitBreakerProvider{
		provider: provider,
		stream:   stream,
		config:   config,
		breakers: make(map[string]*gobreaker.CircuitBreaker),
	}
}

// Chat implements the ProviderPort interface with circuit breaker protection
func (c *CircuitBreakerProvider) Chat(ctx context.Context, req *chat.Request) (*chat.Response, error) {
	if !c.config.Enabled {
		// Pass through if circuit breaker is disabled
		return c.provider.Chat(ctx, req)
	}

	// Extract model from request or use "default" if not specified
	model := c.extractModel(req)
	breaker := c.getOrCreateBreaker(model)

	// Execute the request through the circuit breaker
	result, err := breaker.Execute(func() (interface{}, error) {
		return c.provider.Chat(ctx, req)
	})

	if err != nil {
		// Check if this is a circuit breaker error (circuit open)
		if err == gobreaker.ErrOpenState {
			logrus.WithFields(logrus.Fields{
				"model": model,
				"state": breaker.State(),
			}).Warn("Circuit breaker is open, failing fast")
			return nil, fmt.Errorf("circuit breaker open for model %s: requests are being rejected to prevent cascade failures", model)
		}
		return nil, err
	}

	return result.(*chat.Response), nil
}

// Stream implements the StreamProviderPort interface with circuit breaker protection
func (c *CircuitBreakerProvider) Stream(ctx context.Context, req *chat.Request, onChunk chat.StreamHandler[chat.StreamChunk]) error {
	if !c.config.Enabled {
		// Pass through if circuit breaker is disabled
		return c.stream.Stream(ctx, req, onChunk)
	}

	// Extract model from request or use "default" if not specified
	model := c.extractModel(req)
	breaker := c.getOrCreateBreaker(model)

	// Execute the streaming request through the circuit breaker
	_, err := breaker.Execute(func() (interface{}, error) {
		err := c.stream.Stream(ctx, req, onChunk)
		return nil, err // gobreaker expects a return value, but streaming doesn't have one
	})

	if err != nil {
		// Check if this is a circuit breaker error (circuit open)
		if err == gobreaker.ErrOpenState {
			logrus.WithFields(logrus.Fields{
				"model": model,
				"state": breaker.State(),
			}).Warn("Circuit breaker is open for streaming, failing fast")
			return fmt.Errorf("circuit breaker open for model %s: streaming requests are being rejected to prevent cascade failures", model)
		}
		return err
	}

	return nil
}

// GetCircuitStates returns the current state of all circuit breakers for monitoring
func (c *CircuitBreakerProvider) GetCircuitStates() map[string]gobreaker.State {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	states := make(map[string]gobreaker.State)
	for model, breaker := range c.breakers {
		states[model] = breaker.State()
	}
	return states
}

// getOrCreateBreaker gets or creates a circuit breaker for the specified model
func (c *CircuitBreakerProvider) getOrCreateBreaker(model string) *gobreaker.CircuitBreaker {
	c.mutex.RLock()
	if breaker, exists := c.breakers[model]; exists {
		c.mutex.RUnlock()
		return breaker
	}
	c.mutex.RUnlock()

	// Need to create a new breaker - acquire write lock
	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Double-check pattern: another goroutine might have created it while we waited
	if breaker, exists := c.breakers[model]; exists {
		return breaker
	}

	// Create circuit breaker settings for this model
	settings := gobreaker.Settings{
		Name:        fmt.Sprintf("llm-model-%s", model),
		MaxRequests: c.config.MaxRequests,
		Interval:    0, // No automatic clearing of counts (we rely on timeout)
		Timeout:     c.config.Timeout,
		ReadyToTrip: func(counts gobreaker.Counts) bool {
			// Trip (open) the circuit if we have enough total requests and the failure rate is high
			return counts.Requests >= c.config.FailureThreshold &&
				counts.TotalFailures >= c.config.FailureThreshold
		},
		OnStateChange: func(name string, from gobreaker.State, to gobreaker.State) {
			logrus.WithFields(logrus.Fields{
				"model":      model,
				"from_state": from,
				"to_state":   to,
			}).Info("Circuit breaker state changed")
		},
	}

	breaker := gobreaker.NewCircuitBreaker(settings)
	c.breakers[model] = breaker

	logrus.WithField("model", model).Info("Created new circuit breaker for model")
	return breaker
}

// extractModel extracts the model name from the request
// This handles cases where the model might be specified in the request
func (c *CircuitBreakerProvider) extractModel(req *chat.Request) string {
	if req.Model != "" {
		// Clean up the model name for use as a map key
		// Remove special characters and normalize to lowercase
		model := strings.ToLower(strings.ReplaceAll(req.Model, "/", "-"))
		model = strings.ReplaceAll(model, ".", "-")
		return model
	}
	return "default"
}

// SetContextualRouter sets the contextual router on the underlying provider
// This allows the circuit breaker to work with intelligent model selection
func (c *CircuitBreakerProvider) SetContextualRouter(router bandit.SimilarityRouter) {
	// If the underlying provider supports contextual routing, delegate to it
	if baseProvider, ok := c.provider.(*Provider); ok {
		baseProvider.SetContextualRouter(router)
	}
}
