package openrouter

import (
	"context"
	"encoding/json"
	"llm-router/domain/chat"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewProvider(t *testing.T) {
	apiKey := "test-api-key"
	baseURL := "https://test.openrouter.ai/api/v1"
	models := []string{"anthropic/claude-3.5-sonnet", "openai/gpt-4o"}
	refererURL := "https://test.com"
	appName := "TestApp"

	provider := NewProvider(apiKey, baseURL, models, refererURL, appName)

	assert.NotNil(t, provider)
	assert.Equal(t, apiKey, provider.apiKey)
	assert.Equal(t, baseURL, provider.baseURL)
	assert.Equal(t, models, provider.models)
	assert.Equal(t, refererURL, provider.refererURL)
	assert.Equal(t, appName, provider.appName)
	assert.NotNil(t, provider.httpClient)
	assert.NotNil(t, provider.rng)
	assert.Equal(t, 60*time.Second, provider.httpClient.Timeout)
}

func TestProvider_getRandomModel(t *testing.T) {
	t.Run("with models", func(t *testing.T) {
		models := []string{"model1", "model2", "model3"}
		provider := NewProvider("key", "url", models, "referer", "app")

		// Test multiple times to ensure randomness works
		seenModels := make(map[string]bool)
		for i := 0; i < 50; i++ {
			model := provider.getRandomModel()
			assert.Contains(t, models, model)
			seenModels[model] = true
		}

		// Should see at least one model (very likely to see multiple)
		assert.True(t, len(seenModels) >= 1)
	})

	t.Run("with no models", func(t *testing.T) {
		provider := NewProvider("key", "url", []string{}, "referer", "app")
		model := provider.getRandomModel()
		assert.Empty(t, model)
	})

	t.Run("with single model", func(t *testing.T) {
		models := []string{"single-model"}
		provider := NewProvider("key", "url", models, "referer", "app")

		for i := 0; i < 10; i++ {
			model := provider.getRandomModel()
			assert.Equal(t, "single-model", model)
		}
	})
}

func TestProvider_Chat_Success(t *testing.T) {
	// Create test server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/chat/completions", r.URL.Path)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		assert.Equal(t, "Bearer test-api-key", r.Header.Get("Authorization"))
		assert.Equal(t, "https://test.com", r.Header.Get("HTTP-Referer"))
		assert.Equal(t, "TestApp", r.Header.Get("X-Title"))

		// Verify request body
		var apiReq apiChatRequest
		err := json.NewDecoder(r.Body).Decode(&apiReq)
		require.NoError(t, err)
		assert.Equal(t, "test-model", apiReq.Model)
		assert.False(t, apiReq.Stream)
		assert.Len(t, apiReq.Messages, 1)
		assert.Equal(t, "user", apiReq.Messages[0].Role)
		assert.Equal(t, "Hello", apiReq.Messages[0].Content)

		// Send response
		response := chat.Response{
			ID:      "chatcmpl-123",
			Object:  "chat.completion",
			Created: 1677652288,
			Model:   "test-model",
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
			Usage: chat.Usage{
				PromptTokens:     10,
				CompletionTokens: 5,
				TotalTokens:      15,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewProvider("test-api-key", server.URL, []string{"test-model"}, "https://test.com", "TestApp")

	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "user", Content: "Hello"},
		},
		Stream: false,
	}

	response, err := provider.Chat(context.Background(), req)

	require.NoError(t, err)
	assert.NotNil(t, response)
	assert.Equal(t, "chatcmpl-123", response.ID)
	assert.Equal(t, "chat.completion", response.Object)
	assert.Equal(t, "test-model", response.Model)
	assert.Len(t, response.Choices, 1)
	assert.Equal(t, "assistant", response.Choices[0].Message.Role)
	assert.Equal(t, "Hello there!", response.Choices[0].Message.Content)
}

func TestProvider_Chat_NoModels(t *testing.T) {
	provider := NewProvider("test-api-key", "http://test.com", []string{}, "https://test.com", "TestApp")

	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "user", Content: "Hello"},
		},
		Stream: false,
	}

	response, err := provider.Chat(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, response)
	assert.Contains(t, err.Error(), "no models available")
}

func TestProvider_Chat_ServerError_WithRetry(t *testing.T) {
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount <= 2 {
			// Fail first two attempts
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte("Internal Server Error"))
			return
		}
		// Succeed on third attempt
		response := chat.Response{
			ID:     "chatcmpl-123",
			Object: "chat.completion",
			Model:  "test-model",
			Choices: []chat.Choice{
				{
					Index: 0,
					Message: chat.Message{
						Role:    "assistant",
						Content: "Success after retry!",
					},
					FinishReason: "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := NewProvider("test-api-key", server.URL, []string{"test-model"}, "https://test.com", "TestApp")

	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "user", Content: "Hello"},
		},
		Stream: false,
	}

	response, err := provider.Chat(context.Background(), req)

	require.NoError(t, err)
	assert.NotNil(t, response)
	assert.Equal(t, "Success after retry!", response.Choices[0].Message.Content)
	assert.Equal(t, 3, callCount) // Should have made 3 calls
}

func TestProvider_Chat_PermanentFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("Persistent Server Error"))
	}))
	defer server.Close()

	provider := NewProvider("test-api-key", server.URL, []string{"test-model"}, "https://test.com", "TestApp")

	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "user", Content: "Hello"},
		},
		Stream: false,
	}

	response, err := provider.Chat(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, response)
	assert.Contains(t, err.Error(), "api call failed after 3 attempts")
}

func TestProvider_Chat_NonRetryableError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte("Bad Request"))
	}))
	defer server.Close()

	provider := NewProvider("test-api-key", server.URL, []string{"test-model"}, "https://test.com", "TestApp")

	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "user", Content: "Hello"},
		},
		Stream: false,
	}

	response, err := provider.Chat(context.Background(), req)

	assert.Error(t, err)
	assert.Nil(t, response)
	assert.Contains(t, err.Error(), "status 400")
	assert.Contains(t, err.Error(), "Bad Request")
}

func TestProvider_Stream_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/chat/completions", r.URL.Path)

		// Verify request body
		var apiReq apiChatRequest
		err := json.NewDecoder(r.Body).Decode(&apiReq)
		require.NoError(t, err)
		assert.True(t, apiReq.Stream)
		// Ensure OpenRouter-compliant flag to include usage in final chunk
		require.NotNil(t, apiReq.StreamOptions)
		assert.True(t, apiReq.StreamOptions.IncludeUsage)

		// Send streaming response
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		flusher, ok := w.(http.Flusher)
		require.True(t, ok)

		// Send chunks (last chunk contains usage)
		chunks := []string{
			`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}`,
			`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"test-model","choices":[{"index":0,"delta":{"content":" there!"},"finish_reason":null}]}`,
			`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":7,"total_tokens":19}}`,
			`data: [DONE]`,
		}

		for _, chunk := range chunks {
			w.Write([]byte(chunk + "\n"))
			flusher.Flush()
		}
	}))
	defer server.Close()

	provider := NewProvider("test-api-key", server.URL, []string{"test-model"}, "https://test.com", "TestApp")

	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "user", Content: "Hello"},
		},
		Stream: true,
	}

	var receivedChunks []chat.StreamChunk
	handler := func(chunk chat.StreamChunk) error {
		receivedChunks = append(receivedChunks, chunk)
		return nil
	}

	err := provider.Stream(context.Background(), req, handler)

	require.NoError(t, err)
	assert.Len(t, receivedChunks, 3) // Excludes [DONE]

	// Verify first chunk
	assert.Equal(t, "chatcmpl-123", receivedChunks[0].ID)
	assert.Equal(t, "assistant", receivedChunks[0].Choices[0].Delta.Role)
	assert.Equal(t, "Hello", receivedChunks[0].Choices[0].Delta.Content)

	// Verify second chunk
	assert.Equal(t, " there!", receivedChunks[1].Choices[0].Delta.Content)

	// Verify final chunk
	assert.NotNil(t, receivedChunks[2].Choices[0].FinishReason)
	assert.Equal(t, "stop", *receivedChunks[2].Choices[0].FinishReason)
	assert.Equal(t, 12, receivedChunks[2].Usage.PromptTokens)
	assert.Equal(t, 7, receivedChunks[2].Usage.CompletionTokens)
	assert.Equal(t, 19, receivedChunks[2].Usage.TotalTokens)
}

func TestProvider_Stream_NoModels(t *testing.T) {
	provider := NewProvider("test-api-key", "http://test.com", []string{}, "https://test.com", "TestApp")

	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "user", Content: "Hello"},
		},
		Stream: true,
	}

	handler := func(chunk chat.StreamChunk) error {
		return nil
	}

	err := provider.Stream(context.Background(), req, handler)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no models available")
}

func TestProvider_Stream_ServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("Internal Server Error"))
	}))
	defer server.Close()

	provider := NewProvider("test-api-key", server.URL, []string{"test-model"}, "https://test.com", "TestApp")

	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "user", Content: "Hello"},
		},
		Stream: true,
	}

	handler := func(chunk chat.StreamChunk) error {
		return nil
	}

	err := provider.Stream(context.Background(), req, handler)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "status 500")
}
