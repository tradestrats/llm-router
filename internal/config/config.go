package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"gopkg.in/yaml.v3"
)

// Config represents the complete application configuration
type Config struct {
	Server         ServerConfig         `yaml:"server"`
	LLMProvider    LLMProviderConfig    `yaml:"llm_provider"`
	Bandit         BanditConfig         `yaml:"bandit"`
	Embedding      EmbeddingConfig      `yaml:"embedding"`
	Database       DatabaseConfig       `yaml:"database"`
	Logging        LoggingConfig        `yaml:"logging"`
	CircuitBreaker CircuitBreakerConfig `yaml:"circuit_breaker"`
}

type ServerConfig struct {
	Host        string   `yaml:"host"`
	Port        string   `yaml:"port"`
	AppName     string   `yaml:"app_name"`
	RefererURL  string   `yaml:"referer_url"`
	CorsOrigins []string `yaml:"cors_origins"`
}

type LLMProviderConfig struct {
	APIKey        string   `yaml:"api_key"`
	BaseURL       string   `yaml:"base_url"`
	AllowedModels []string `yaml:"allowed_models"`
}

type BanditConfig struct {
	DefaultModel     string                 `yaml:"default_model"`
	ThompsonSampling ThompsonSamplingConfig `yaml:"thompson_sampling"`
	Similarity       SimilarityConfig       `yaml:"similarity"`
	ColdStart        ColdStartConfig        `yaml:"cold_start"`
	Persistence      PersistenceConfig      `yaml:"persistence"`
}

type ThompsonSamplingConfig struct {
	FeedbackWeight  float64 `yaml:"feedback_weight"`
	LatencyWeight   float64 `yaml:"latency_weight"`
	CostWeight      float64 `yaml:"cost_weight"`
	ExplorationRate float64 `yaml:"exploration_rate"`
}

type SimilarityConfig struct {
	Threshold          float64 `yaml:"threshold"`
	MaxSimilarRequests int     `yaml:"max_similar_requests"`
	RecencyDays        int     `yaml:"recency_days"`
	MinSimilarRequests int     `yaml:"min_similar_requests"`
}

type ColdStartConfig struct {
	MinConfidenceScore   float64 `yaml:"min_confidence_score"`
	OptimisticPrior      float64 `yaml:"optimistic_prior"`
	ExplorationBonus     float64 `yaml:"exploration_bonus"`
	MinRequestsForGlobal int     `yaml:"min_requests_for_global"`
}

type PersistenceConfig struct {
	BatchSize int `yaml:"batch_size"`
}

type EmbeddingConfig struct {
	ServiceType string `yaml:"service_type"`
	ServiceURL  string `yaml:"service_url"`
	ModelPath   string `yaml:"model_path"`
	MaxWorkers  int    `yaml:"max_workers"`
	CacheSize   int    `yaml:"cache_size"`
	TimeoutMs   int    `yaml:"timeout_ms"`
}

type DatabaseConfig struct {
	EnablePersistence bool   `yaml:"enable_persistence"`
	URL               string `yaml:"url"`
	Host              string `yaml:"host"`
	Port              string `yaml:"port"`
	User              string `yaml:"user"`
	Password          string `yaml:"password"`
	Name              string `yaml:"name"`
	SSLMode           string `yaml:"ssl_mode"`
	Workers           int    `yaml:"workers"`
	BufferSize        int    `yaml:"buffer_size"`
}

type LoggingConfig struct {
	Level        string `yaml:"level"`
	Format       string `yaml:"format"`
	ReportCaller bool   `yaml:"report_caller"`
}

type CircuitBreakerConfig struct {
	Enabled          bool          `yaml:"enabled"`
	FailureThreshold uint32        `yaml:"failure_threshold"`
	SuccessThreshold uint32        `yaml:"success_threshold"`
	Timeout          time.Duration `yaml:"timeout"`
	MaxRequests      uint32        `yaml:"max_requests"`
}

// LoadYAML loads configuration from YAML file with environment variable overrides
func LoadYAML(configPath string) (*Config, error) {
	// Set default config path if not provided
	if configPath == "" {
		configPath = "config.yaml"
	}

	config := &Config{}

	// Load YAML file if it exists
	if _, err := os.Stat(configPath); err == nil {
		yamlFile, err := os.ReadFile(configPath)
		if err != nil {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}

		// Expand environment variables in YAML content
		expandedYAML := os.ExpandEnv(string(yamlFile))

		if err := yaml.Unmarshal([]byte(expandedYAML), config); err != nil {
			return nil, fmt.Errorf("failed to parse config file: %w", err)
		}

		logrus.WithField("config_file", configPath).Info("Loaded configuration from YAML file")
	} else {
		logrus.WithField("config_file", configPath).Warn("Config file not found, using defaults and environment variables")
		config = getDefaultConfig()
	}

	// Apply environment variable overrides
	config = applyEnvironmentOverrides(config)

	// Validate configuration
	if err := validateConfig(config); err != nil {
		return nil, fmt.Errorf("configuration validation failed: %w", err)
	}

	return config, nil
}

// getDefaultConfig returns a configuration with sensible defaults
func getDefaultConfig() *Config {
	return &Config{
		Server: ServerConfig{
			Host:        "0.0.0.0",
			Port:        "8080",
			AppName:     "LLM Router",
			RefererURL:  "https://llm-router.ai",
			CorsOrigins: []string{"*"},
		},
		LLMProvider: LLMProviderConfig{
			BaseURL: "https://openrouter.ai/api/v1",
			AllowedModels: []string{
				// Free models from OpenRouter
				"meta-llama/llama-3.2-3b-instruct:free",
				"meta-llama/llama-3.2-1b-instruct:free",
				"google/gemma-2-9b-it:free",
				"microsoft/phi-3-mini-128k-instruct:free",
				"microsoft/phi-3-medium-128k-instruct:free",
				"qwen/qwen-2-7b-instruct:free",
				"huggingfaceh4/zephyr-7b-beta:free",
				"openchat/openchat-7b:free",
				// Premium models (require credits)
				"anthropic/claude-3.5-sonnet",
				"openai/gpt-4o",
			},
		},
		Bandit: BanditConfig{
			DefaultModel: "meta-llama/llama-3.2-3b-instruct:free",
			ThompsonSampling: ThompsonSamplingConfig{
				FeedbackWeight:  0.6,
				LatencyWeight:   0.2, // Positive because normalizedLatency rewards speed
				CostWeight:      -0.2,
				ExplorationRate: 0.15, // Slightly higher exploration for better learning
			},
			Similarity: SimilarityConfig{
				Threshold:          0.7,
				MaxSimilarRequests: 50,
				RecencyDays:        30,
				MinSimilarRequests: 5,
			},
			ColdStart: ColdStartConfig{
				MinConfidenceScore:   0.1,
				OptimisticPrior:      0.8,
				ExplorationBonus:     0.1,
				MinRequestsForGlobal: 10,
			},
			Persistence: PersistenceConfig{
				BatchSize: 100,
			},
		},
		Embedding: EmbeddingConfig{
			ServiceType: "http",
			ServiceURL:  "http://localhost:8001",
			ModelPath:   "./models/all-MiniLM-L6-v2",
			MaxWorkers:  4,
			CacheSize:   1000,
			TimeoutMs:   5000,
		},
		Database: DatabaseConfig{
			EnablePersistence: false, // Start with in-memory mode for easier setup
			Host:              "localhost",
			Port:              "5432",
			User:              "llm-router",
			Name:              "llm-router",
			SSLMode:           "disable",
			Workers:           5,
			BufferSize:        1000,
		},
		Logging: LoggingConfig{
			Level:        "info",
			Format:       "auto",
			ReportCaller: false,
		},
		CircuitBreaker: CircuitBreakerConfig{
			Enabled:          true,
			FailureThreshold: 5,
			SuccessThreshold: 2,
			Timeout:          60 * time.Second,
			MaxRequests:      3,
		},
	}
}

// applyEnvironmentOverrides applies environment variable overrides to config
func applyEnvironmentOverrides(config *Config) *Config {
	// Server overrides
	if val := os.Getenv("HOST"); val != "" {
		config.Server.Host = val
	}
	if val := os.Getenv("PORT"); val != "" {
		config.Server.Port = val
	}
	if val := os.Getenv("APP_NAME"); val != "" {
		config.Server.AppName = val
	}
	if val := os.Getenv("REFERER_URL"); val != "" {
		config.Server.RefererURL = val
	}
	if val := os.Getenv("CORS_ORIGINS"); val != "" {
		config.Server.CorsOrigins = strings.Split(val, ",")
		for i := range config.Server.CorsOrigins {
			config.Server.CorsOrigins[i] = strings.TrimSpace(config.Server.CorsOrigins[i])
		}
	}

	// LLM Provider overrides
	if val := os.Getenv("OPENROUTER_API_KEY"); val != "" {
		config.LLMProvider.APIKey = val
	}
	if val := os.Getenv("OPENROUTER_BASE_URL"); val != "" {
		config.LLMProvider.BaseURL = val
	}
	if val := os.Getenv("LLM_MODELS"); val != "" {
		models := strings.Split(val, ",")
		for i := range models {
			models[i] = strings.TrimSpace(models[i])
		}
		config.LLMProvider.AllowedModels = models
	}

	// Bandit overrides
	if val := os.Getenv("DEFAULT_MODEL"); val != "" {
		config.Bandit.DefaultModel = val
	}
	if val := os.Getenv("FEEDBACK_WEIGHT"); val != "" {
		if f, err := strconv.ParseFloat(val, 64); err == nil {
			config.Bandit.ThompsonSampling.FeedbackWeight = f
		}
	}
	if val := os.Getenv("LATENCY_WEIGHT"); val != "" {
		if f, err := strconv.ParseFloat(val, 64); err == nil {
			config.Bandit.ThompsonSampling.LatencyWeight = f
		}
	}
	if val := os.Getenv("COST_WEIGHT"); val != "" {
		if f, err := strconv.ParseFloat(val, 64); err == nil {
			config.Bandit.ThompsonSampling.CostWeight = f
		}
	}
	if val := os.Getenv("EXPLORATION_RATE"); val != "" {
		if f, err := strconv.ParseFloat(val, 64); err == nil {
			config.Bandit.ThompsonSampling.ExplorationRate = f
		}
	}
	if val := os.Getenv("SIMILARITY_THRESHOLD"); val != "" {
		if f, err := strconv.ParseFloat(val, 64); err == nil {
			config.Bandit.Similarity.Threshold = f
		}
	}
	if val := os.Getenv("MAX_SIMILAR_REQUESTS"); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			config.Bandit.Similarity.MaxSimilarRequests = i
		}
	}
	if val := os.Getenv("BATCH_SIZE"); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			config.Bandit.Persistence.BatchSize = i
		}
	}

	// Embedding service overrides
	if val := os.Getenv("EMBEDDING_SERVICE_TYPE"); val != "" {
		config.Embedding.ServiceType = val
	}
	if val := os.Getenv("EMBEDDING_SERVICE_URL"); val != "" {
		config.Embedding.ServiceURL = val
	}
	if val := os.Getenv("EMBEDDING_MODEL_PATH"); val != "" {
		config.Embedding.ModelPath = val
	}
	if val := os.Getenv("EMBEDDING_MAX_WORKERS"); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			config.Embedding.MaxWorkers = i
		}
	}
	if val := os.Getenv("EMBEDDING_CACHE_SIZE"); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			config.Embedding.CacheSize = i
		}
	}
	if val := os.Getenv("EMBEDDING_TIMEOUT_MS"); val != "" {
		if i, err := strconv.Atoi(val); err == nil {
			config.Embedding.TimeoutMs = i
		}
	}

	// Database overrides
	if val := os.Getenv("ENABLE_PERSISTENCE"); val != "" {
		if b, err := strconv.ParseBool(val); err == nil {
			config.Database.EnablePersistence = b
		}
	}
	if val := os.Getenv("DATABASE_URL"); val != "" {
		config.Database.URL = val
	}
	if val := os.Getenv("DATABASE_HOST"); val != "" {
		config.Database.Host = val
	}
	if val := os.Getenv("DATABASE_PORT"); val != "" {
		config.Database.Port = val
	}
	if val := os.Getenv("DATABASE_USER"); val != "" {
		config.Database.User = val
	}
	if val := os.Getenv("DATABASE_PASSWORD"); val != "" {
		config.Database.Password = val
	}
	if val := os.Getenv("DATABASE_NAME"); val != "" {
		config.Database.Name = val
	}
	if val := os.Getenv("DATABASE_SSL_MODE"); val != "" {
		config.Database.SSLMode = val
	}

	// Logging overrides
	if val := os.Getenv("LOG_LEVEL"); val != "" {
		config.Logging.Level = val
	}
	if val := os.Getenv("LOG_FORMAT"); val != "" {
		config.Logging.Format = val
	}
	if val := os.Getenv("LOG_REPORT_CALLER"); val != "" {
		if b, err := strconv.ParseBool(val); err == nil {
			config.Logging.ReportCaller = b
		}
	}

	// Circuit breaker overrides
	if val := os.Getenv("CIRCUIT_BREAKER_ENABLED"); val != "" {
		if b, err := strconv.ParseBool(val); err == nil {
			config.CircuitBreaker.Enabled = b
		}
	}
	if val := os.Getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD"); val != "" {
		if i, err := strconv.ParseUint(val, 10, 32); err == nil {
			config.CircuitBreaker.FailureThreshold = uint32(i)
		}
	}
	if val := os.Getenv("CIRCUIT_BREAKER_SUCCESS_THRESHOLD"); val != "" {
		if i, err := strconv.ParseUint(val, 10, 32); err == nil {
			config.CircuitBreaker.SuccessThreshold = uint32(i)
		}
	}
	if val := os.Getenv("CIRCUIT_BREAKER_TIMEOUT"); val != "" {
		if d, err := time.ParseDuration(val); err == nil {
			config.CircuitBreaker.Timeout = d
		}
	}
	if val := os.Getenv("CIRCUIT_BREAKER_MAX_REQUESTS"); val != "" {
		if i, err := strconv.ParseUint(val, 10, 32); err == nil {
			config.CircuitBreaker.MaxRequests = uint32(i)
		}
	}

	return config
}

// validateConfig validates the configuration and returns errors for invalid values
func validateConfig(config *Config) error {
	var errors []string

	// Validate required fields
	if config.LLMProvider.APIKey == "" {
		errors = append(errors, "OPENROUTER_API_KEY is required - get one from https://openrouter.ai/keys")
	}

	if len(config.LLMProvider.AllowedModels) == 0 {
		errors = append(errors, "at least one allowed model must be specified in LLM_MODELS or config.yaml")
	}

	// Validate bandit parameters
	if config.Bandit.ThompsonSampling.FeedbackWeight < 0 || config.Bandit.ThompsonSampling.FeedbackWeight > 1 {
		errors = append(errors, fmt.Sprintf("FEEDBACK_WEIGHT must be between 0 and 1 (current: %.2f)", config.Bandit.ThompsonSampling.FeedbackWeight))
	}

	if config.Bandit.ThompsonSampling.LatencyWeight < 0 {
		errors = append(errors, fmt.Sprintf("LATENCY_WEIGHT should be positive - higher values favor faster models (current: %.2f)", config.Bandit.ThompsonSampling.LatencyWeight))
	}

	if config.Bandit.ThompsonSampling.CostWeight > 0 {
		errors = append(errors, fmt.Sprintf("COST_WEIGHT should be negative - more negative values favor cheaper models (current: %.2f)", config.Bandit.ThompsonSampling.CostWeight))
	}

	if config.Bandit.Similarity.Threshold < 0 || config.Bandit.Similarity.Threshold > 1 {
		errors = append(errors, fmt.Sprintf("SIMILARITY_THRESHOLD must be between 0 and 1 (current: %.2f)", config.Bandit.Similarity.Threshold))
	}

	if config.Bandit.ColdStart.OptimisticPrior < 0 || config.Bandit.ColdStart.OptimisticPrior > 1 {
		errors = append(errors, fmt.Sprintf("optimistic_prior must be between 0 and 1 (current: %.2f)", config.Bandit.ColdStart.OptimisticPrior))
	}

	// Validate model format (warn but don't fail)
	for _, model := range config.LLMProvider.AllowedModels {
		if !strings.Contains(model, "/") {
			logrus.WithField("model", model).Warn("Model may not be valid - expected format: provider/model")
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("configuration validation errors: %s", strings.Join(errors, "; "))
	}

	return nil
}

// GetDatabaseDSN constructs the database connection string
func (c *Config) GetDatabaseDSN() string {
	if c.Database.URL != "" {
		return c.Database.URL
	}

	return fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=%s",
		c.Database.Host,
		c.Database.Port,
		c.Database.User,
		c.Database.Password,
		c.Database.Name,
		c.Database.SSLMode,
	)
}

// Backward compatibility function
func Load() (*Config, error) {
	return LoadYAML("")
}
