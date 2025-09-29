package chat

import (
	"context"
	"errors"
	"testing"
	"time"

	"llm-router/domain/chat"
	"llm-router/domain/persistence"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// Mock provider that implements both Chat and Stream interfaces
type MockChatProvider struct {
	mock.Mock
}

func (m *MockChatProvider) Chat(ctx context.Context, req *chat.Request) (*chat.Response, error) {
	args := m.Called(ctx, req)
	return args.Get(0).(*chat.Response), args.Error(1)
}

func (m *MockChatProvider) Stream(ctx context.Context, req *chat.Request, onChunk chat.StreamHandler[chat.StreamChunk]) error {
	args := m.Called(ctx, req, onChunk)
	return args.Error(0)
}

// Mock request tracker for testing
type MockRequestTracker struct {
	mock.Mock
}

func (m *MockRequestTracker) StartTracking(ctx context.Context, requestID uuid.UUID, requestData []byte, model string, isStreaming bool) error {
	args := m.Called(ctx, requestID, requestData, model, isStreaming)
	return args.Error(0)
}

func (m *MockRequestTracker) CompleteTracking(ctx context.Context, requestID uuid.UUID, responseData []byte, metrics persistence.RequestMetrics) error {
	args := m.Called(ctx, requestID, responseData, metrics)
	return args.Error(0)
}

func (m *MockRequestTracker) FailTracking(ctx context.Context, requestID uuid.UUID, errorMsg string) error {
	args := m.Called(ctx, requestID, errorMsg)
	return args.Error(0)
}

func (m *MockRequestTracker) SubmitFeedback(ctx context.Context, requestID uuid.UUID, feedbackText string, score float64) error {
	args := m.Called(ctx, requestID, feedbackText, score)
	return args.Error(0)
}

func (m *MockRequestTracker) UpdateEmbedding(ctx context.Context, requestID uuid.UUID, embedding []float32) error {
	args := m.Called(ctx, requestID, embedding)
	return args.Error(0)
}

func TestService_ChatWithTracking_Success(t *testing.T) {
	provider := &MockChatProvider{}
	tracker := &MockRequestTracker{}

	service := NewService(provider, provider, tracker, nil, nil) // nil embedding service and bandit router for test

	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "user", Content: "Hello, world!"},
		},
		Stream: false,
	}

	expectedResponse := &chat.Response{
		ID:      "test-id",
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   "test-model",
		Choices: []chat.Choice{
			{
				Index:        0,
				Message:      chat.Message{Role: "assistant", Content: "Hello! How can I help you?"},
				FinishReason: "stop",
			},
		},
		Usage: chat.Usage{
			PromptTokens:     10,
			CompletionTokens: 15,
			TotalTokens:      25,
		},
	}

	// Setup mock expectations
	provider.On("Chat", mock.Anything, req).Return(expectedResponse, nil)
	tracker.On("StartTracking", mock.Anything, mock.AnythingOfType("uuid.UUID"), mock.Anything, "test-model", false).Return(nil)
	tracker.On("CompleteTracking", mock.Anything, mock.AnythingOfType("uuid.UUID"), mock.Anything, mock.AnythingOfType("persistence.RequestMetrics")).Return(nil)

	// Execute
	ctx := context.Background()
	resp, err := service.Chat(ctx, req)

	// Assert
	assert.NoError(t, err)
	assert.Equal(t, expectedResponse, resp)

	// Give time for async tracking to complete
	time.Sleep(50 * time.Millisecond)

	// Verify all expectations were met
	provider.AssertExpectations(t)
	tracker.AssertExpectations(t)
}

func TestService_ChatWithTracking_Failure(t *testing.T) {
	provider := &MockChatProvider{}
	tracker := &MockRequestTracker{}

	service := NewService(provider, provider, tracker, nil, nil) // nil embedding service and bandit router for test

	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "user", Content: "Hello, world!"},
		},
		Stream: false,
	}

	expectedError := errors.New("provider error")

	// Setup mock expectations
	provider.On("Chat", mock.Anything, req).Return(&chat.Response{}, expectedError)
	tracker.On("StartTracking", mock.Anything, mock.AnythingOfType("uuid.UUID"), mock.Anything, "unknown", false).Return(nil)
	tracker.On("FailTracking", mock.Anything, mock.AnythingOfType("uuid.UUID"), "provider error").Return(nil)

	// Execute
	ctx := context.Background()
	resp, err := service.Chat(ctx, req)

	// Assert
	assert.Error(t, err)
	assert.Equal(t, expectedError, err)
	assert.Nil(t, resp)

	// Give time for async tracking to complete
	time.Sleep(50 * time.Millisecond)

	// Verify all expectations were met
	provider.AssertExpectations(t)
	tracker.AssertExpectations(t)
}

func TestService_ChatWithoutTracking_Success(t *testing.T) {
	provider := &MockChatProvider{}

	// Create service without tracker
	service := NewServiceWithoutTracking(provider, provider)

	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "user", Content: "Hello, world!"},
		},
		Stream: false,
	}

	expectedResponse := &chat.Response{
		ID:      "test-id",
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   "test-model",
		Choices: []chat.Choice{
			{
				Index:        0,
				Message:      chat.Message{Role: "assistant", Content: "Hello! How can I help you?"},
				FinishReason: "stop",
			},
		},
		Usage: chat.Usage{
			PromptTokens:     10,
			CompletionTokens: 15,
			TotalTokens:      25,
		},
	}

	// Setup mock expectations (only provider, no tracker)
	provider.On("Chat", mock.Anything, req).Return(expectedResponse, nil)

	// Execute
	ctx := context.Background()
	resp, err := service.Chat(ctx, req)

	// Assert
	assert.NoError(t, err)
	assert.Equal(t, expectedResponse, resp)

	// Verify expectations
	provider.AssertExpectations(t)
}

func TestService_StreamWithTracking_Success(t *testing.T) {
	provider := &MockChatProvider{}
	tracker := &MockRequestTracker{}

	service := NewService(provider, provider, tracker, nil, nil) // nil embedding service and bandit router for test

	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "user", Content: "Hello, world!"},
		},
		Stream: true,
	}

	// Create test chunks
	chunk1 := chat.StreamChunk{
		ID:      "test-id",
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   "test-model",
		Choices: []chat.StreamChoice{
			{
				Index: 0,
				Delta: chat.StreamDelta{Content: "Hello"},
			},
		},
	}

	chunk2 := chat.StreamChunk{
		ID:      "test-id",
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   "test-model",
		Choices: []chat.StreamChoice{
			{
				Index:        0,
				Delta:        chat.StreamDelta{},
				FinishReason: stringPtr("stop"),
			},
		},
		Usage: &chat.Usage{
			PromptTokens:     10,
			CompletionTokens: 15,
			TotalTokens:      25,
		},
	}

	// Mock stream provider to call onChunk with test chunks
	provider.On("Stream", mock.Anything, req, mock.AnythingOfType("chat.StreamHandler[llm-router/domain/chat.StreamChunk]")).
		Run(func(args mock.Arguments) {
			handler := args.Get(2).(chat.StreamHandler[chat.StreamChunk])
			handler(chunk1)
			handler(chunk2)
		}).Return(nil)

	tracker.On("StartTracking", mock.Anything, mock.AnythingOfType("uuid.UUID"), mock.Anything, "test-model", true).Return(nil)
	tracker.On("CompleteTracking", mock.Anything, mock.AnythingOfType("uuid.UUID"), mock.Anything, mock.AnythingOfType("persistence.RequestMetrics")).Return(nil)

	// Execute
	ctx := context.Background()
	var receivedChunks []chat.StreamChunk

	err := service.Stream(ctx, req, func(chunk chat.StreamChunk) error {
		receivedChunks = append(receivedChunks, chunk)
		return nil
	})

	// Assert
	assert.NoError(t, err)
	assert.Len(t, receivedChunks, 2)
	assert.Equal(t, chunk1, receivedChunks[0])
	assert.Equal(t, chunk2, receivedChunks[1])

	// Give time for async tracking to complete
	time.Sleep(50 * time.Millisecond)

	// Verify expectations
	provider.AssertExpectations(t)
	tracker.AssertExpectations(t)
}

func TestService_CalculateCost(t *testing.T) {
	provider := &MockChatProvider{}
	tracker := &MockRequestTracker{}

	service := NewService(provider, provider, tracker, nil, nil) // nil embedding service and bandit router for test

	tests := []struct {
		name     string
		model    string
		usage    chat.Usage
		expected float64
	}{
		{
			name:  "Claude 3.5 Sonnet",
			model: "anthropic/claude-3.5-sonnet",
			usage: chat.Usage{
				PromptTokens:     1000,
				CompletionTokens: 500,
				TotalTokens:      1500,
			},
			// Cost calculation now returns 0.0 as placeholder - usage metrics are tracked separately
			expected: 0.0,
		},
		{
			name:  "GPT-4o",
			model: "openai/gpt-4o",
			usage: chat.Usage{
				PromptTokens:     1000,
				CompletionTokens: 500,
				TotalTokens:      1500,
			},
			// Cost calculation now returns 0.0 as placeholder - usage metrics are tracked separately
			expected: 0.0,
		},
		{
			name:  "Unknown Model",
			model: "unknown/model",
			usage: chat.Usage{
				PromptTokens:     1000,
				CompletionTokens: 500,
				TotalTokens:      1500,
			},
			// Cost calculation now returns 0.0 as placeholder - usage metrics are tracked separately
			expected: 0.0,
		},
		{
			name:  "Model with cost from OpenRouter",
			model: "anthropic/claude-3.5-sonnet",
			usage: chat.Usage{
				PromptTokens:     1000,
				CompletionTokens: 500,
				TotalTokens:      1500,
				Cost:             &[]float64{0.0125}[0], // Cost provided by OpenRouter
			},
			// Should use cost from OpenRouter response
			expected: 0.0125,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cost := service.calculateCost(tt.model, tt.usage)
			assert.InDelta(t, tt.expected, cost, 0.0001) // Allow small floating point differences
		})
	}
}

func TestService_ValidationWithTracking(t *testing.T) {
	provider := &MockChatProvider{}
	tracker := &MockRequestTracker{}

	service := NewService(provider, provider, tracker, nil, nil) // nil embedding service and bandit router for test

	ctx := context.Background()

	// Test empty messages
	req := &chat.Request{
		Messages: []chat.Message{},
		Stream:   false,
	}

	_, err := service.Chat(ctx, req)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "messages cannot be empty")

	// Test streaming flag validation
	req = &chat.Request{
		Messages: []chat.Message{
			{Role: "user", Content: "Hello"},
		},
		Stream: true,
	}

	_, err = service.Chat(ctx, req)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "use Stream for streaming requests")

	// Test too many messages
	messages := make([]chat.Message, 101)
	for i := range messages {
		messages[i] = chat.Message{Role: "user", Content: "test"}
	}

	req = &chat.Request{
		Messages: messages,
		Stream:   false,
	}

	_, err = service.Chat(ctx, req)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "too many messages")
}

// Helper function to create string pointers
func stringPtr(s string) *string {
	return &s
}
