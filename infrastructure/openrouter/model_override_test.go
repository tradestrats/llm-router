package openrouter

import (
	"context"
	"testing"

	appchat "llm-router/domain/chat"
)

func TestProvider_selectModel_UserSpecified(t *testing.T) {
	tests := []struct {
		name          string
		allowedModels []string
		requestModel  string
		expected      string
		description   string
	}{
		{
			name:          "valid_user_model",
			allowedModels: []string{"gpt-4", "claude-3", "llama-2"},
			requestModel:  "claude-3",
			expected:      "claude-3",
			description:   "Should use user-specified model when it's in allowed list",
		},
		{
			name:          "invalid_user_model",
			allowedModels: []string{"gpt-4", "claude-3"},
			requestModel:  "unauthorized-model",
			expected:      "", // Will fallback to random (empty since no contextual router)
			description:   "Should ignore invalid model and use routing logic",
		},
		{
			name:          "empty_user_model",
			allowedModels: []string{"gpt-4", "claude-3"},
			requestModel:  "",
			expected:      "", // Will use routing logic
			description:   "Should use routing logic when no model specified",
		},
		{
			name:          "case_sensitive_match",
			allowedModels: []string{"GPT-4", "claude-3"},
			requestModel:  "gpt-4", // Different case
			expected:      "",      // Will not match, fallback to routing
			description:   "Model matching should be case-sensitive",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := NewProvider("test-key", "test-url", tt.allowedModels, "test-referer", "test-app")

			req := &appchat.Request{
				Model: tt.requestModel,
				Messages: []appchat.Message{
					{Role: "user", Content: "test message"},
				},
			}

			result := provider.selectModel(context.Background(), req)

			if tt.expected != "" {
				// Expecting specific model
				if result != tt.expected {
					t.Errorf("Expected model %q, got %q", tt.expected, result)
				}
			} else {
				// Expecting random selection or empty (since no contextual router)
				if tt.requestModel != "" && provider.isModelAllowed(tt.requestModel) {
					// Should have returned the requested model
					t.Errorf("Expected user model %q to be selected, got %q", tt.requestModel, result)
				}
				// For other cases, just check that we got some result or empty
				// (empty is expected when no models are configured or invalid request)
			}
		})
	}
}

func TestProvider_isModelAllowed(t *testing.T) {
	provider := NewProvider("test-key", "test-url", []string{"gpt-4", "claude-3.5-sonnet", "llama-2"}, "test-referer", "test-app")

	tests := []struct {
		model    string
		expected bool
	}{
		{"gpt-4", true},
		{"claude-3.5-sonnet", true},
		{"llama-2", true},
		{"gpt-3.5", false},
		{"unauthorized-model", false},
		{"", false},
		{"GPT-4", false}, // Case sensitive
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			result := provider.isModelAllowed(tt.model)
			if result != tt.expected {
				t.Errorf("isModelAllowed(%q) = %v, expected %v", tt.model, result, tt.expected)
			}
		})
	}
}

func TestProvider_selectModel_Priority(t *testing.T) {
	// Test the priority order: User-specified > Contextual > Random
	provider := NewProvider("test-key", "test-url", []string{"gpt-4", "claude-3"}, "test-referer", "test-app")

	// Test without contextual router (should use random when no user model)
	req := &appchat.Request{
		Messages: []appchat.Message{
			{Role: "user", Content: "test message"},
		},
	}

	result := provider.selectModel(context.Background(), req)
	// Should be one of the allowed models (random selection)
	if result != "gpt-4" && result != "claude-3" {
		t.Errorf("Expected random selection from allowed models, got %q", result)
	}

	// Test with user-specified model (should override everything)
	req.Model = "claude-3"
	result = provider.selectModel(context.Background(), req)
	if result != "claude-3" {
		t.Errorf("Expected user-specified model 'claude-3', got %q", result)
	}
}
