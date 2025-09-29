package chat

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMessage_JSONMarshaling(t *testing.T) {
	tests := []struct {
		name     string
		message  Message
		expected string
	}{
		{
			name: "basic message",
			message: Message{
				Role:    "user",
				Content: "Hello, world!",
			},
			expected: `{"role":"user","content":"Hello, world!"}`,
		},
		{
			name: "assistant message",
			message: Message{
				Role:    "assistant",
				Content: "I'm here to help!",
			},
			expected: `{"role":"assistant","content":"I'm here to help!"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.message)
			require.NoError(t, err)
			assert.JSONEq(t, tt.expected, string(data))

			// Test unmarshaling
			var unmarshaled Message
			err = json.Unmarshal(data, &unmarshaled)
			require.NoError(t, err)
			assert.Equal(t, tt.message, unmarshaled)
		})
	}
}

func TestRequest_JSONMarshaling(t *testing.T) {
	request := Request{
		Messages: []Message{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi there!"},
		},
		Stream: true,
	}

	data, err := json.Marshal(request)
	require.NoError(t, err)

	var unmarshaled Request
	err = json.Unmarshal(data, &unmarshaled)
	require.NoError(t, err)
	assert.Equal(t, request, unmarshaled)
}

func TestResponse_JSONMarshaling(t *testing.T) {
	response := Response{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: 1677652288,
		Model:   "gpt-3.5-turbo",
		Choices: []Choice{
			{
				Index: 0,
				Message: Message{
					Role:    "assistant",
					Content: "Hello there!",
				},
				FinishReason: "stop",
			},
		},
		Usage: Usage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}

	data, err := json.Marshal(response)
	require.NoError(t, err)

	var unmarshaled Response
	err = json.Unmarshal(data, &unmarshaled)
	require.NoError(t, err)
	assert.Equal(t, response, unmarshaled)
}

func TestStreamChunk_JSONMarshaling(t *testing.T) {
	chunk := StreamChunk{
		ID:      "chatcmpl-123",
		Object:  "chat.completion.chunk",
		Created: 1677652288,
		Model:   "gpt-3.5-turbo",
		Choices: []StreamChoice{
			{
				Index: 0,
				Delta: StreamDelta{
					Role:    "assistant",
					Content: "Hello",
				},
				FinishReason: nil,
			},
		},
	}

	data, err := json.Marshal(chunk)
	require.NoError(t, err)

	var unmarshaled StreamChunk
	err = json.Unmarshal(data, &unmarshaled)
	require.NoError(t, err)
	assert.Equal(t, chunk, unmarshaled)
}

func TestErrorResponse_JSONMarshaling(t *testing.T) {
	errorResp := ErrorResponse{
		Error: "Invalid API key",
	}

	data, err := json.Marshal(errorResp)
	require.NoError(t, err)

	expected := `{"error":"Invalid API key"}`
	assert.JSONEq(t, expected, string(data))

	var unmarshaled ErrorResponse
	err = json.Unmarshal(data, &unmarshaled)
	require.NoError(t, err)
	assert.Equal(t, errorResp, unmarshaled)
}
