package openrouter

import (
	"context"
	"errors"
	"testing"
	"time"

	"llm-router/domain/bandit"
	"llm-router/domain/chat"

	"github.com/sony/gobreaker"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// MockProvider is a mock implementation of the chat provider interface
type MockProvider struct {
	mock.Mock
}

func (m *MockProvider) Chat(ctx context.Context, req *chat.Request) (*chat.Response, error) {
	args := m.Called(ctx, req)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*chat.Response), args.Error(1)
}

// MockStreamProvider is a mock implementation of the streaming provider interface
type MockStreamProvider struct {
	mock.Mock
}

func (m *MockStreamProvider) Stream(ctx context.Context, req *chat.Request, onChunk chat.StreamHandler[chat.StreamChunk]) error {
	args := m.Called(ctx, req, onChunk)
	return args.Error(0)
}

func TestNewCircuitBreakerProvider(t *testing.T) {
	mockProvider := &MockProvider{}
	mockStream := &MockStreamProvider{}

	config := DefaultCircuitBreakerConfig()

	cbProvider := NewCircuitBreakerProvider(mockProvider, mockStream, config)

	assert.NotNil(t, cbProvider)
	assert.Equal(t, config, cbProvider.config)
	assert.Equal(t, mockProvider, cbProvider.provider)
	assert.Equal(t, mockStream, cbProvider.stream)
	assert.NotNil(t, cbProvider.breakers)
}

func TestNewCircuitBreakerProvider_Disabled(t *testing.T) {
	mockProvider := &MockProvider{}
	mockStream := &MockStreamProvider{}

	config := CircuitBreakerConfig{
		Enabled: false,
	}

	cbProvider := NewCircuitBreakerProvider(mockProvider, mockStream, config)

	assert.NotNil(t, cbProvider)
	assert.False(t, cbProvider.config.Enabled)
}

func TestCircuitBreakerProvider_Chat_Success(t *testing.T) {
	mockProvider := &MockProvider{}
	mockStream := &MockStreamProvider{}

	config := CircuitBreakerConfig{
		Enabled:          true,
		FailureThreshold: 3,
		SuccessThreshold: 2,
		Timeout:          30 * time.Second,
		MaxRequests:      2,
	}

	cbProvider := NewCircuitBreakerProvider(mockProvider, mockStream, config)

	req := &chat.Request{
		Model: "test-model",
		Messages: []chat.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	expectedResponse := &chat.Response{
		ID:      "test-response",
		Model:   "test-model",
		Choices: []chat.Choice{{Message: chat.Message{Role: "assistant", Content: "Hi there!"}}},
	}

	mockProvider.On("Chat", mock.Anything, req).Return(expectedResponse, nil)

	ctx := context.Background()
	response, err := cbProvider.Chat(ctx, req)

	assert.NoError(t, err)
	assert.Equal(t, expectedResponse, response)
	mockProvider.AssertExpectations(t)
}

func TestCircuitBreakerProvider_Chat_Disabled(t *testing.T) {
	mockProvider := &MockProvider{}
	mockStream := &MockStreamProvider{}

	config := CircuitBreakerConfig{
		Enabled: false,
	}

	cbProvider := NewCircuitBreakerProvider(mockProvider, mockStream, config)

	req := &chat.Request{
		Model: "test-model",
		Messages: []chat.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	expectedResponse := &chat.Response{
		ID:    "test-response",
		Model: "test-model",
	}

	mockProvider.On("Chat", mock.Anything, req).Return(expectedResponse, nil)

	ctx := context.Background()
	response, err := cbProvider.Chat(ctx, req)

	assert.NoError(t, err)
	assert.Equal(t, expectedResponse, response)
	mockProvider.AssertExpectations(t)
}

func TestCircuitBreakerProvider_Chat_CircuitOpen(t *testing.T) {
	mockProvider := &MockProvider{}
	mockStream := &MockStreamProvider{}

	config := CircuitBreakerConfig{
		Enabled:          true,
		FailureThreshold: 2, // Low threshold for faster testing
		SuccessThreshold: 1,
		Timeout:          1 * time.Second,
		MaxRequests:      1,
	}

	cbProvider := NewCircuitBreakerProvider(mockProvider, mockStream, config)

	req := &chat.Request{
		Model: "test-model",
		Messages: []chat.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	// First, cause failures to open the circuit
	testError := errors.New("service unavailable")
	mockProvider.On("Chat", mock.Anything, req).Return(nil, testError).Times(2)

	ctx := context.Background()

	// Make calls that will fail and eventually open the circuit
	for i := 0; i < 2; i++ {
		_, err := cbProvider.Chat(ctx, req)
		assert.Error(t, err)
	}

	// Now the circuit should be open, and we should get a circuit breaker error
	_, err := cbProvider.Chat(ctx, req)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "circuit breaker open")

	mockProvider.AssertExpectations(t)
}

func TestCircuitBreakerProvider_Stream_Success(t *testing.T) {
	mockProvider := &MockProvider{}
	mockStream := &MockStreamProvider{}

	config := DefaultCircuitBreakerConfig()
	cbProvider := NewCircuitBreakerProvider(mockProvider, mockStream, config)

	req := &chat.Request{
		Model: "test-model",
		Messages: []chat.Message{
			{Role: "user", Content: "Hello"},
		},
		Stream: true,
	}

	onChunk := func(chunk chat.StreamChunk) error {
		return nil
	}

	mockStream.On("Stream", mock.Anything, req, mock.Anything).Return(nil)

	ctx := context.Background()
	err := cbProvider.Stream(ctx, req, onChunk)

	assert.NoError(t, err)
	mockStream.AssertExpectations(t)
}

func TestCircuitBreakerProvider_Stream_Disabled(t *testing.T) {
	mockProvider := &MockProvider{}
	mockStream := &MockStreamProvider{}

	config := CircuitBreakerConfig{
		Enabled: false,
	}

	cbProvider := NewCircuitBreakerProvider(mockProvider, mockStream, config)

	req := &chat.Request{
		Model:  "test-model",
		Stream: true,
	}

	onChunk := func(chunk chat.StreamChunk) error {
		return nil
	}

	mockStream.On("Stream", mock.Anything, req, mock.Anything).Return(nil)

	ctx := context.Background()
	err := cbProvider.Stream(ctx, req, onChunk)

	assert.NoError(t, err)
	mockStream.AssertExpectations(t)
}

func TestCircuitBreakerProvider_GetCircuitStates(t *testing.T) {
	mockProvider := &MockProvider{}
	mockStream := &MockStreamProvider{}

	config := DefaultCircuitBreakerConfig()
	cbProvider := NewCircuitBreakerProvider(mockProvider, mockStream, config)

	// Initially, no circuit breakers should exist
	states := cbProvider.GetCircuitStates()
	assert.Empty(t, states)

	// Make a request to create a circuit breaker
	req := &chat.Request{
		Model: "test-model",
		Messages: []chat.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	mockProvider.On("Chat", mock.Anything, req).Return(&chat.Response{}, nil)

	ctx := context.Background()
	_, _ = cbProvider.Chat(ctx, req)

	// Now we should have one circuit breaker
	states = cbProvider.GetCircuitStates()
	assert.Len(t, states, 1)
	assert.Contains(t, states, "test-model")
	assert.Equal(t, gobreaker.StateClosed, states["test-model"])

	mockProvider.AssertExpectations(t)
}

func TestCircuitBreakerProvider_ExtractModel(t *testing.T) {
	mockProvider := &MockProvider{}
	mockStream := &MockStreamProvider{}

	config := DefaultCircuitBreakerConfig()
	cbProvider := NewCircuitBreakerProvider(mockProvider, mockStream, config)

	tests := []struct {
		name     string
		request  *chat.Request
		expected string
	}{
		{
			name: "model with slashes",
			request: &chat.Request{
				Model: "openai/gpt-4",
			},
			expected: "openai-gpt-4",
		},
		{
			name: "model with dots",
			request: &chat.Request{
				Model: "claude-3.5-sonnet",
			},
			expected: "claude-3-5-sonnet",
		},
		{
			name: "empty model",
			request: &chat.Request{
				Model: "",
			},
			expected: "default",
		},
		{
			name: "complex model name",
			request: &chat.Request{
				Model: "anthropic/claude-3.5-sonnet@20241022",
			},
			expected: "anthropic-claude-3-5-sonnet@20241022",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := cbProvider.extractModel(tt.request)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestCircuitBreakerProvider_SetContextualRouter(t *testing.T) {
	// Create a real provider to test the delegation
	realProvider := &Provider{}
	mockStream := &MockStreamProvider{}

	config := DefaultCircuitBreakerConfig()
	cbProvider := NewCircuitBreakerProvider(realProvider, mockStream, config)

	// Mock bandit router
	mockRouter := &MockBanditRouter{}

	// Set the contextual router
	cbProvider.SetContextualRouter(mockRouter)

	// Verify that the underlying provider received the router
	assert.Equal(t, mockRouter, realProvider.contextualRouter)
}

// MockBanditRouter is a mock implementation for testing
type MockBanditRouter struct {
	mock.Mock
}

func (m *MockBanditRouter) SelectModel(ctx context.Context, req *chat.Request) (string, error) {
	args := m.Called(ctx, req)
	return args.String(0), args.Error(1)
}

func (m *MockBanditRouter) UpdatePerformance(ctx context.Context, requestID string, model string, metrics bandit.PerformanceMetrics) error {
	args := m.Called(ctx, requestID, model, metrics)
	return args.Error(0)
}

func (m *MockBanditRouter) Health(ctx context.Context) error {
	args := m.Called(ctx)
	return args.Error(0)
}

func (m *MockBanditRouter) Close() error {
	args := m.Called()
	return args.Error(0)
}

func TestDefaultCircuitBreakerConfig(t *testing.T) {
	config := DefaultCircuitBreakerConfig()

	assert.True(t, config.Enabled)
	assert.Equal(t, uint32(5), config.FailureThreshold)
	assert.Equal(t, uint32(2), config.SuccessThreshold)
	assert.Equal(t, 60*time.Second, config.Timeout)
	assert.Equal(t, uint32(3), config.MaxRequests)
}
