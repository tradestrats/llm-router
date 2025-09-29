package chat

import (
	"context"
	"errors"
	"llm-router/domain/chat"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// Mock implementations for testing
type MockProvider struct {
	mock.Mock
}

func (m *MockProvider) Chat(ctx context.Context, req *chat.Request) (*chat.Response, error) {
	args := m.Called(req)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*chat.Response), args.Error(1)
}

type MockStreamProvider struct {
	mock.Mock
}

func (m *MockStreamProvider) Stream(ctx context.Context, req *chat.Request, onChunk chat.StreamHandler[chat.StreamChunk]) error {
	args := m.Called(req, onChunk)
	return args.Error(0)
}

func TestNewServiceWithoutTracking(t *testing.T) {
	provider := &MockProvider{}
	streamProvider := &MockStreamProvider{}

	service := NewServiceWithoutTracking(provider, streamProvider)

	assert.NotNil(t, service)
	assert.Equal(t, provider, service.provider)
	assert.Equal(t, streamProvider, service.stream)
	assert.Nil(t, service.tracker) // Should be nil for service without tracking
}

func TestService_Chat_ValidRequest(t *testing.T) {
	provider := &MockProvider{}
	streamProvider := &MockStreamProvider{}
	service := NewServiceWithoutTracking(provider, streamProvider)

	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "user", Content: "Hello"},
		},
		Stream: false,
	}

	expectedResponse := &chat.Response{
		ID:     "test-id",
		Object: "chat.completion",
		Model:  "test-model",
		Choices: []chat.Choice{
			{
				Index: 0,
				Message: chat.Message{
					Role:    "assistant",
					Content: "Hello there!",
				},
				FinishReason: "stop",
			},
		},
	}

	provider.On("Chat", req).Return(expectedResponse, nil)

	response, err := service.Chat(context.Background(), req)

	assert.NoError(t, err)
	assert.Equal(t, expectedResponse, response)
	provider.AssertExpectations(t)
}

func TestService_Chat_EmptyMessages(t *testing.T) {
	provider := &MockProvider{}
	streamProvider := &MockStreamProvider{}
	service := NewServiceWithoutTracking(provider, streamProvider)

	req := &chat.Request{
		Messages: []chat.Message{},
		Stream:   false,
	}

	response, err := service.Chat(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, response)
	assert.Equal(t, "messages cannot be empty", err.Error())
}

func TestService_Chat_StreamRequested(t *testing.T) {
	provider := &MockProvider{}
	streamProvider := &MockStreamProvider{}
	service := NewServiceWithoutTracking(provider, streamProvider)

	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "user", Content: "Hello"},
		},
		Stream: true,
	}

	response, err := service.Chat(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, response)
	assert.Equal(t, "use Stream for streaming requests", err.Error())
}

func TestService_Chat_TooManyMessages(t *testing.T) {
	provider := &MockProvider{}
	streamProvider := &MockStreamProvider{}
	service := NewServiceWithoutTracking(provider, streamProvider)

	// Create 101 messages (exceeds max of 100)
	messages := make([]chat.Message, 101)
	for i := 0; i < 101; i++ {
		messages[i] = chat.Message{Role: "user", Content: "test"}
	}

	req := &chat.Request{
		Messages: messages,
		Stream:   false,
	}

	response, err := service.Chat(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, response)
	assert.Contains(t, err.Error(), "too many messages")
}

func TestService_Chat_InvalidMessageRole(t *testing.T) {
	provider := &MockProvider{}
	streamProvider := &MockStreamProvider{}
	service := NewServiceWithoutTracking(provider, streamProvider)

	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "invalid", Content: "Hello"},
		},
		Stream: false,
	}

	response, err := service.Chat(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, response)
	assert.Contains(t, err.Error(), "invalid role")
}

func TestService_Chat_EmptyRole(t *testing.T) {
	provider := &MockProvider{}
	streamProvider := &MockStreamProvider{}
	service := NewServiceWithoutTracking(provider, streamProvider)

	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "", Content: "Hello"},
		},
		Stream: false,
	}

	response, err := service.Chat(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, response)
	assert.Contains(t, err.Error(), "role cannot be empty")
}

func TestService_Chat_EmptyContent(t *testing.T) {
	provider := &MockProvider{}
	streamProvider := &MockStreamProvider{}
	service := NewServiceWithoutTracking(provider, streamProvider)

	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "user", Content: ""},
		},
		Stream: false,
	}

	response, err := service.Chat(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, response)
	assert.Contains(t, err.Error(), "content cannot be empty")
}

func TestService_Chat_ContentTooLong(t *testing.T) {
	provider := &MockProvider{}
	streamProvider := &MockStreamProvider{}
	service := NewServiceWithoutTracking(provider, streamProvider)

	longContent := strings.Repeat("a", 50001) // Exceeds max of 50000

	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "user", Content: longContent},
		},
		Stream: false,
	}

	response, err := service.Chat(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, response)
	assert.Contains(t, err.Error(), "content too long")
}

func TestService_Chat_ProviderError(t *testing.T) {
	provider := &MockProvider{}
	streamProvider := &MockStreamProvider{}
	service := NewServiceWithoutTracking(provider, streamProvider)

	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "user", Content: "Hello"},
		},
		Stream: false,
	}

	expectedError := errors.New("provider error")
	provider.On("Chat", req).Return(nil, expectedError)

	response, err := service.Chat(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, response)
	assert.Equal(t, expectedError, err)
	provider.AssertExpectations(t)
}

func TestService_Stream_ValidRequest(t *testing.T) {
	provider := &MockProvider{}
	streamProvider := &MockStreamProvider{}
	service := NewServiceWithoutTracking(provider, streamProvider)

	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "user", Content: "Hello"},
		},
		Stream: true,
	}

	var capturedHandler chat.StreamHandler[chat.StreamChunk]
	streamProvider.On("Stream", req, mock.AnythingOfType("chat.StreamHandler[llm-router/domain/chat.StreamChunk]")).
		Run(func(args mock.Arguments) {
			capturedHandler = args.Get(1).(chat.StreamHandler[chat.StreamChunk])
		}).Return(nil)

	handler := func(chunk chat.StreamChunk) error {
		return nil
	}

	err := service.Stream(context.Background(), req, handler)

	assert.NoError(t, err)
	assert.NotNil(t, capturedHandler)
	streamProvider.AssertExpectations(t)
}

func TestService_Stream_EmptyMessages(t *testing.T) {
	provider := &MockProvider{}
	streamProvider := &MockStreamProvider{}
	service := NewServiceWithoutTracking(provider, streamProvider)

	req := &chat.Request{
		Messages: []chat.Message{},
		Stream:   true,
	}

	handler := func(chunk chat.StreamChunk) error {
		return nil
	}

	err := service.Stream(context.Background(), req, handler)

	assert.Error(t, err)
	assert.Equal(t, "messages cannot be empty", err.Error())
}

func TestService_Stream_StreamNotRequested(t *testing.T) {
	provider := &MockProvider{}
	streamProvider := &MockStreamProvider{}
	service := NewServiceWithoutTracking(provider, streamProvider)

	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "user", Content: "Hello"},
		},
		Stream: false,
	}

	handler := func(chunk chat.StreamChunk) error {
		return nil
	}

	err := service.Stream(context.Background(), req, handler)

	assert.Error(t, err)
	assert.Equal(t, "set stream=true for streaming", err.Error())
}

func TestService_Stream_ValidationErrors(t *testing.T) {
	provider := &MockProvider{}
	streamProvider := &MockStreamProvider{}
	service := NewServiceWithoutTracking(provider, streamProvider)

	handler := func(chunk chat.StreamChunk) error {
		return nil
	}

	tests := []struct {
		name        string
		request     *chat.Request
		expectedErr string
	}{
		{
			name: "too many messages",
			request: &chat.Request{
				Messages: make([]chat.Message, 101),
				Stream:   true,
			},
			expectedErr: "too many messages",
		},
		{
			name: "empty role",
			request: &chat.Request{
				Messages: []chat.Message{
					{Role: "", Content: "Hello"},
				},
				Stream: true,
			},
			expectedErr: "role cannot be empty",
		},
		{
			name: "empty content",
			request: &chat.Request{
				Messages: []chat.Message{
					{Role: "user", Content: ""},
				},
				Stream: true,
			},
			expectedErr: "content cannot be empty",
		},
		{
			name: "invalid role",
			request: &chat.Request{
				Messages: []chat.Message{
					{Role: "invalid", Content: "Hello"},
				},
				Stream: true,
			},
			expectedErr: "invalid role",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Fill messages with valid data for "too many messages" test
			if tt.name == "too many messages" {
				for i := range tt.request.Messages {
					tt.request.Messages[i] = chat.Message{Role: "user", Content: "test"}
				}
			}

			err := service.Stream(context.Background(), tt.request, handler)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), tt.expectedErr)
		})
	}
}
