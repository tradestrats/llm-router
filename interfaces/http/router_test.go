package httpiface

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	domain "llm-router/domain/chat"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// Mock service for testing
type MockChatService struct {
	mock.Mock
}

func (m *MockChatService) Chat(ctx context.Context, req *domain.Request) (*domain.Response, error) {
	args := m.Called(req)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*domain.Response), args.Error(1)
}

func (m *MockChatService) Stream(ctx context.Context, req *domain.Request, onChunk domain.StreamHandler[domain.StreamChunk]) error {
	args := m.Called(req, onChunk)
	return args.Error(0)
}

func TestNewRouter(t *testing.T) {
	service := &MockChatService{}
	corsOrigins := []string{"https://example.com", "https://test.com"}

	router := NewRouter(service, corsOrigins)

	assert.NotNil(t, router)
	assert.Equal(t, service, router.service)
	assert.Equal(t, corsOrigins, router.corsOrigins)
}

func TestRouter_SetupRoutes(t *testing.T) {
	service := &MockChatService{}
	router := NewRouter(service, []string{"*"})

	engine := router.SetupRoutes()

	assert.NotNil(t, engine)

	// Test that routes are registered
	routes := engine.Routes()
	routePaths := make([]string, len(routes))
	for i, route := range routes {
		routePaths[i] = route.Path
	}

	assert.Contains(t, routePaths, "/health")
	assert.Contains(t, routePaths, "/chat/completions")
	assert.Contains(t, routePaths, "/live")
	assert.Contains(t, routePaths, "/ready")
}

func TestRouter_healthCheck(t *testing.T) {
	service := &MockChatService{}
	router := NewRouter(service, []string{"*"})
	engine := router.SetupRoutes()

	req, _ := http.NewRequest("GET", "/health", nil)
	req.Header.Set("X-Request-ID", "550e8400-e29b-41d4-a716-446655440000")
	w := httptest.NewRecorder()

	engine.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)

	assert.Equal(t, "healthy", response["status"])
	assert.Equal(t, "llm-proxy-router", response["service"])
	assert.Equal(t, "1.0.0", response["version"])
	assert.NotEmpty(t, response["timestamp"])

	checks, ok := response["checks"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "ok", checks["api"])
}

func TestRouter_chatCompletions_Success(t *testing.T) {
	service := &MockChatService{}
	router := NewRouter(service, []string{"*"})
	engine := router.SetupRoutes()

	requestBody := domain.Request{
		Messages: []domain.Message{
			{Role: "user", Content: "Hello"},
		},
		Stream: false,
	}

	expectedResponse := &domain.Response{
		ID:     "chatcmpl-123",
		Object: "chat.completion",
		Model:  "test-model",
		Choices: []domain.Choice{
			{
				Index: 0,
				Message: domain.Message{
					Role:    "assistant",
					Content: "Hello there!",
				},
				FinishReason: "stop",
			},
		},
	}

	service.On("Chat", &requestBody).Return(expectedResponse, nil)

	jsonData, _ := json.Marshal(requestBody)
	req, _ := http.NewRequest("POST", "/chat/completions", bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Request-ID", "550e8400-e29b-41d4-a716-446655440000")

	w := httptest.NewRecorder()
	engine.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response domain.Response
	err := json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)
	assert.Equal(t, *expectedResponse, response)

	service.AssertExpectations(t)
}

func TestRouter_chatCompletions_EmptyMessages(t *testing.T) {
	service := &MockChatService{}
	router := NewRouter(service, []string{"*"})
	engine := router.SetupRoutes()

	requestBody := domain.Request{
		Messages: []domain.Message{},
		Stream:   false,
	}

	jsonData, _ := json.Marshal(requestBody)
	req, _ := http.NewRequest("POST", "/chat/completions", bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Request-ID", "550e8400-e29b-41d4-a716-446655440000")

	w := httptest.NewRecorder()
	engine.ServeHTTP(w, req)

	assert.Equal(t, http.StatusBadRequest, w.Code)

	var response domain.ErrorResponse
	err := json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)
	assert.Equal(t, "Messages cannot be empty", response.Error)
}

func TestRouter_chatCompletions_InvalidJSON(t *testing.T) {
	service := &MockChatService{}
	router := NewRouter(service, []string{"*"})
	engine := router.SetupRoutes()

	req, _ := http.NewRequest("POST", "/chat/completions", bytes.NewBufferString("invalid json"))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Request-ID", "550e8400-e29b-41d4-a716-446655440000")

	w := httptest.NewRecorder()
	engine.ServeHTTP(w, req)

	assert.Equal(t, http.StatusBadRequest, w.Code)

	var response domain.ErrorResponse
	err := json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)
	assert.Equal(t, "Invalid request format", response.Error)
}

func TestRouter_chatCompletions_ServiceError(t *testing.T) {
	service := &MockChatService{}
	router := NewRouter(service, []string{"*"})
	engine := router.SetupRoutes()

	requestBody := domain.Request{
		Messages: []domain.Message{
			{Role: "user", Content: "Hello"},
		},
		Stream: false,
	}

	service.On("Chat", &requestBody).Return(nil, assert.AnError)

	jsonData, _ := json.Marshal(requestBody)
	req, _ := http.NewRequest("POST", "/chat/completions", bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Request-ID", "550e8400-e29b-41d4-a716-446655440000")

	w := httptest.NewRecorder()
	engine.ServeHTTP(w, req)

	assert.Equal(t, http.StatusInternalServerError, w.Code)

	var response domain.ErrorResponse
	err := json.Unmarshal(w.Body.Bytes(), &response)
	require.NoError(t, err)
	assert.Equal(t, "Failed to process request", response.Error)

	service.AssertExpectations(t)
}

func TestRouter_chatCompletions_Streaming(t *testing.T) {
	service := &MockChatService{}
	router := NewRouter(service, []string{"*"})
	engine := router.SetupRoutes()

	requestBody := domain.Request{
		Messages: []domain.Message{
			{Role: "user", Content: "Hello"},
		},
		Stream: true,
	}

	// Mock the streaming behavior
	service.On("Stream", &requestBody, mock.MatchedBy(func(handler domain.StreamHandler[domain.StreamChunk]) bool {
		return true
	})).
		Run(func(args mock.Arguments) {
			handler := args.Get(1).(domain.StreamHandler[domain.StreamChunk])

			// Simulate sending chunks
			chunks := []domain.StreamChunk{
				{
					ID:     "chatcmpl-123",
					Object: "chat.completion.chunk",
					Model:  "test-model",
					Choices: []domain.StreamChoice{
						{
							Index: 0,
							Delta: domain.StreamDelta{
								Role:    "assistant",
								Content: "Hello",
							},
							FinishReason: nil,
						},
					},
				},
				{
					ID:     "chatcmpl-123",
					Object: "chat.completion.chunk",
					Model:  "test-model",
					Choices: []domain.StreamChoice{
						{
							Index: 0,
							Delta: domain.StreamDelta{
								Content: " there!",
							},
							FinishReason: stringPtr("stop"),
						},
					},
				},
			}

			for _, chunk := range chunks {
				handler(chunk)
			}
		}).Return(nil)

	jsonData, _ := json.Marshal(requestBody)
	req, _ := http.NewRequest("POST", "/chat/completions", bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Request-ID", "550e8400-e29b-41d4-a716-446655440000")

	w := httptest.NewRecorder()
	engine.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "text/event-stream", w.Header().Get("Content-Type"))
	assert.Equal(t, "no-cache", w.Header().Get("Cache-Control"))
	assert.Equal(t, "keep-alive", w.Header().Get("Connection"))

	responseBody := w.Body.String()
	assert.Contains(t, responseBody, "data: ")
	assert.Contains(t, responseBody, "data: [DONE]")

	service.AssertExpectations(t)
}

func TestRouter_corsMiddleware(t *testing.T) {
	service := &MockChatService{}
	corsOrigins := []string{"https://example.com", "https://test.com"}
	router := NewRouter(service, corsOrigins)
	engine := router.SetupRoutes()

	req, _ := http.NewRequest("GET", "/health", nil)
	req.Header.Set("X-Request-ID", "550e8400-e29b-41d4-a716-446655440000")
	w := httptest.NewRecorder()

	engine.ServeHTTP(w, req)

	assert.Equal(t, "https://example.com, https://test.com", w.Header().Get("Access-Control-Allow-Origin"))
	assert.Equal(t, "GET, POST, OPTIONS", w.Header().Get("Access-Control-Allow-Methods"))
	assert.Equal(t, "Content-Type, Authorization", w.Header().Get("Access-Control-Allow-Headers"))
}

func TestRouter_corsMiddleware_OPTIONS(t *testing.T) {
	service := &MockChatService{}
	router := NewRouter(service, []string{"*"})
	engine := router.SetupRoutes()

	req, _ := http.NewRequest("OPTIONS", "/chat/completions", nil)
	w := httptest.NewRecorder()

	engine.ServeHTTP(w, req)

	assert.Equal(t, http.StatusNoContent, w.Code)
	assert.Equal(t, "*", w.Header().Get("Access-Control-Allow-Origin"))
}

func TestRouter_requestIDMiddleware(t *testing.T) {
	service := &MockChatService{}
	router := NewRouter(service, []string{"*"})
	engine := router.SetupRoutes()

	// Test with client-provided request ID
	req, _ := http.NewRequest("GET", "/health", nil)
	req.Header.Set("X-Request-ID", "550e8400-e29b-41d4-a716-446655440000")
	w := httptest.NewRecorder()

	engine.ServeHTTP(w, req)

	// Should echo back the client-provided ID
	requestID := w.Header().Get("X-Request-ID")
	assert.Equal(t, "550e8400-e29b-41d4-a716-446655440000", requestID)
	
	// Should also have UUID header
	requestUUID := w.Header().Get("X-Request-UUID")
	assert.Equal(t, "550e8400-e29b-41d4-a716-446655440000", requestUUID)

	// Test without client-provided request ID - should be rejected
	req2, _ := http.NewRequest("GET", "/health", nil)
	w2 := httptest.NewRecorder()

	engine.ServeHTTP(w2, req2)

	// Should return 400 Bad Request
	assert.Equal(t, http.StatusBadRequest, w2.Code)
	
	// Should have error message
	var response map[string]interface{}
	err := json.Unmarshal(w2.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.Equal(t, "Missing required header: X-Request-ID or X-Correlation-ID", response["error"])
	assert.Equal(t, "All requests must include a request ID for proper tracking and feedback persistence", response["message"])
}

// Helper function to create string pointer
func stringPtr(s string) *string {
	return &s
}
