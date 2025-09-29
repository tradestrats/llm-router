package config

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// getEnv is a helper function for testing environment variable handling
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func TestGetEnv(t *testing.T) {
	tests := []struct {
		name         string
		key          string
		defaultValue string
		envValue     string
		expected     string
	}{
		{
			name:         "environment variable set",
			key:          "TEST_KEY",
			defaultValue: "default",
			envValue:     "custom",
			expected:     "custom",
		},
		{
			name:         "environment variable not set",
			key:          "TEST_KEY_NOT_SET",
			defaultValue: "default",
			envValue:     "",
			expected:     "default",
		},
		{
			name:         "empty environment variable",
			key:          "TEST_KEY_EMPTY",
			defaultValue: "default",
			envValue:     "",
			expected:     "default",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Clean up environment
			os.Unsetenv(tt.key)

			if tt.envValue != "" {
				os.Setenv(tt.key, tt.envValue)
				defer os.Unsetenv(tt.key)
			}

			result := getEnv(tt.key, tt.defaultValue)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestLoad_DefaultValues(t *testing.T) {
	// Clear relevant environment variables
	envVars := []string{
		"PORT", "HOST", "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL",
		"LLM_MODELS", "LOG_LEVEL", "CORS_ORIGINS", "REFERER_URL", "APP_NAME",
	}

	for _, envVar := range envVars {
		os.Unsetenv(envVar)
	}

	// Set required API key
	os.Setenv("OPENROUTER_API_KEY", "test-api-key")
	defer os.Unsetenv("OPENROUTER_API_KEY")

	config, err := Load()
	require.NoError(t, err)

	assert.Equal(t, "8080", config.Server.Port)
	assert.Equal(t, "0.0.0.0", config.Server.Host)
	assert.Equal(t, "test-api-key", config.LLMProvider.APIKey)
	assert.Equal(t, "https://openrouter.ai/api/v1", config.LLMProvider.BaseURL)
	assert.Equal(t, "info", config.Logging.Level)
	assert.Equal(t, "https://llm-router.ai", config.Server.RefererURL)
	assert.Equal(t, "LLM Router", config.Server.AppName)
	assert.Equal(t, []string{"meta-llama/llama-3.2-3b-instruct:free", "meta-llama/llama-3.2-1b-instruct:free", "google/gemma-2-9b-it:free", "microsoft/phi-3-mini-128k-instruct:free", "microsoft/phi-3-medium-128k-instruct:free", "qwen/qwen-2-7b-instruct:free", "huggingfaceh4/zephyr-7b-beta:free", "openchat/openchat-7b:free", "anthropic/claude-3.5-sonnet", "openai/gpt-4o"}, config.LLMProvider.AllowedModels)
	assert.Equal(t, []string{"*"}, config.Server.CorsOrigins)
}

func TestLoad_CustomValues(t *testing.T) {
	// Set custom environment variables
	envVars := map[string]string{
		"PORT":                "3000",
		"HOST":                "localhost",
		"OPENROUTER_API_KEY":  "custom-api-key",
		"OPENROUTER_BASE_URL": "https://custom.openrouter.ai/api/v1",
		"LLM_MODELS":          "anthropic/claude-3.5-sonnet, openai/gpt-4o-mini,   google/gemini-pro",
		"LOG_LEVEL":           "debug",
		"CORS_ORIGINS":        "https://example.com, https://test.com,   https://dev.com",
		"REFERER_URL":         "https://custom.com",
		"APP_NAME":            "CustomApp",
	}

	for key, value := range envVars {
		os.Setenv(key, value)
		defer os.Unsetenv(key)
	}

	config, err := Load()
	require.NoError(t, err)

	assert.Equal(t, "3000", config.Server.Port)
	assert.Equal(t, "localhost", config.Server.Host)
	assert.Equal(t, "custom-api-key", config.LLMProvider.APIKey)
	assert.Equal(t, "https://custom.openrouter.ai/api/v1", config.LLMProvider.BaseURL)
	assert.Equal(t, "debug", config.Logging.Level)
	assert.Equal(t, "https://custom.com", config.Server.RefererURL)
	assert.Equal(t, "CustomApp", config.Server.AppName)

	expectedModels := []string{"anthropic/claude-3.5-sonnet", "openai/gpt-4o-mini", "google/gemini-pro"}
	assert.Equal(t, expectedModels, config.LLMProvider.AllowedModels)

	expectedOrigins := []string{"https://example.com", "https://test.com", "https://dev.com"}
	assert.Equal(t, expectedOrigins, config.Server.CorsOrigins)
}

func TestLoad_LLMModelsValidation(t *testing.T) {
	tests := []struct {
		name          string
		modelsEnvVar  string
		apiKey        string
		shouldSucceed bool
		description   string
	}{
		{
			name:          "valid models",
			modelsEnvVar:  "anthropic/claude-3.5-sonnet,openai/gpt-4o",
			apiKey:        "test-key",
			shouldSucceed: true,
			description:   "should succeed with valid models",
		},
		{
			name:          "single model",
			modelsEnvVar:  "anthropic/claude-3.5-sonnet",
			apiKey:        "test-key",
			shouldSucceed: true,
			description:   "should succeed with single model",
		},
		{
			name:          "empty models",
			modelsEnvVar:  "",
			apiKey:        "test-key",
			shouldSucceed: false,
			description:   "should fail with empty models",
		},
		{
			name:          "whitespace only models",
			modelsEnvVar:  "   ,   ",
			apiKey:        "test-key",
			shouldSucceed: false,
			description:   "should fail with whitespace-only models",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Clear environment
			os.Unsetenv("LLM_MODELS")
			os.Unsetenv("OPENROUTER_API_KEY")

			if tt.modelsEnvVar != "" {
				os.Setenv("LLM_MODELS", tt.modelsEnvVar)
				defer os.Unsetenv("LLM_MODELS")
			}

			os.Setenv("OPENROUTER_API_KEY", tt.apiKey)
			defer os.Unsetenv("OPENROUTER_API_KEY")

			if tt.shouldSucceed {
				config, err := Load()
				assert.NoError(t, err, tt.description)
				assert.NotNil(t, config)
				assert.NotEmpty(t, config.LLMProvider.AllowedModels)
			} else {
				// Since the function calls logrus.Fatal, we can't easily test the failure case
				// without mocking logrus or changing the implementation
				// For now, we'll skip the failure test cases
				t.Skip("Cannot test logrus.Fatal cases without mocking")
			}
		})
	}
}

func TestLoad_APIKeyValidation(t *testing.T) {
	// Clear API key
	os.Unsetenv("OPENROUTER_API_KEY")

	// This would call logrus.Fatal, so we can't easily test it
	// without mocking logrus or changing the implementation
	t.Skip("Cannot test logrus.Fatal cases without mocking")
}

func TestLoad_ModelFormatWarning(t *testing.T) {
	// Set model without slash (should trigger warning)
	os.Setenv("OPENROUTER_API_KEY", "test-key")
	os.Setenv("LLM_MODELS", "invalid-model-name")
	defer os.Unsetenv("OPENROUTER_API_KEY")
	defer os.Unsetenv("LLM_MODELS")

	config, err := Load()

	// Should still succeed but would log a warning
	assert.NoError(t, err)
	assert.NotNil(t, config)
	assert.Equal(t, []string{"invalid-model-name"}, config.LLMProvider.AllowedModels)
}
