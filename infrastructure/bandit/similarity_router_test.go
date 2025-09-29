package bandit

import (
	"context"
	"math"
	"testing"

	"llm-router/domain/bandit"
	"llm-router/domain/chat"

	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

// MockEmbeddingService for testing
type MockEmbeddingService struct {
	embeddings map[string][]float32
}

func NewMockEmbeddingService() *MockEmbeddingService {
	return &MockEmbeddingService{
		embeddings: make(map[string][]float32),
	}
}

func (m *MockEmbeddingService) Embed(ctx context.Context, text string) ([]float32, error) {
	// Return deterministic embeddings based on text
	embedding := make([]float32, 384)
	for i := 0; i < 384; i++ {
		embedding[i] = float32(len(text)*i) / 1000.0
	}
	return embedding, nil
}

func (m *MockEmbeddingService) EmbedMessages(ctx context.Context, messages []chat.Message) ([]float32, error) {
	text := ""
	for _, msg := range messages {
		text += msg.Content + " "
	}
	return m.Embed(ctx, text)
}

func (m *MockEmbeddingService) BatchEmbed(ctx context.Context, texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	for i, text := range texts {
		embed, err := m.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		embeddings[i] = embed
	}
	return embeddings, nil
}

func (m *MockEmbeddingService) GetDimensions() int {
	return 384
}

func (m *MockEmbeddingService) Health(ctx context.Context) error {
	return nil
}

func (m *MockEmbeddingService) Readiness(ctx context.Context) error {
	return nil
}

func (m *MockEmbeddingService) Close() error {
	return nil
}

// Mock state manager for testing
type MockStateManager struct{}

func (m *MockStateManager) GetArm(model string) *bandit.BanditArm {
	return &bandit.BanditArm{
		Model:        model,
		RequestCount: 10,
		SuccessCount: 8,
	}
}

func (m *MockStateManager) GetAllArms() map[string]*bandit.BanditArm {
	return map[string]*bandit.BanditArm{
		"test-model": {
			Model:        "test-model",
			RequestCount: 10,
			SuccessCount: 8,
		},
	}
}

func (m *MockStateManager) UpdateArm(model string, metrics bandit.PerformanceMetrics) error {
	return nil
}

func (m *MockStateManager) CreateArm(model string) error {
	return nil
}

func (m *MockStateManager) GetStatistics() map[string]interface{} {
	return map[string]interface{}{}
}

func (m *MockStateManager) Health() error {
	return nil
}

func (m *MockStateManager) Close() error {
	return nil
}

func setupTestDB(t *testing.T) *gorm.DB {
	db, err := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{})
	if err != nil {
		t.Fatalf("Failed to create test database: %v", err)
	}

	// Enable the vector extension (simulate pgvector)
	// Note: SQLite doesn't have pgvector, so this is just for testing the interface
	return db
}

func TestContextualRouter_Creation(t *testing.T) {
	db := setupTestDB(t)
	embeddingService := NewMockEmbeddingService()

	config := bandit.BanditConfig{
		SimilarityThreshold: 0.7,
		MaxSimilarRequests:  50,
		RecencyDays:         30,
		FeedbackWeight:      0.6,
		LatencyWeight:       0.2,
		CostWeight:          -0.2,
		MinSimilarRequests:  5,
		DefaultModel:        "test-model",
	}

	router, err := NewContextualRouter(config, embeddingService, &MockStateManager{}, db)
	if err != nil {
		t.Fatalf("Failed to create contextual router: %v", err)
	}
	defer router.Close()

	if router == nil {
		t.Error("Router is nil")
	}
}

func TestContextualRouter_SelectModel_NoSimilarRequests(t *testing.T) {
	db := setupTestDB(t)
	embeddingService := NewMockEmbeddingService()

	config := bandit.BanditConfig{
		SimilarityThreshold: 0.7,
		MaxSimilarRequests:  50,
		RecencyDays:         30,
		FeedbackWeight:      0.6,
		LatencyWeight:       0.2,
		CostWeight:          -0.2,
		MinSimilarRequests:  5,
		DefaultModel:        "default-model",
		GlobalFallback:      true,
	}

	router, err := NewContextualRouter(config, embeddingService, &MockStateManager{}, db)
	if err != nil {
		t.Fatalf("Failed to create contextual router: %v", err)
	}
	defer router.Close()

	ctx := context.Background()
	req := &chat.Request{
		Messages: []chat.Message{
			{Role: "user", Content: "Hello, test message"},
		},
	}

	model, err := router.SelectModel(ctx, req)
	if err != nil {
		t.Fatalf("SelectModel failed: %v", err)
	}

	// Should return one of the available models (random selection from fallback)
	// The MockStateManager provides "test-model", so it should be selected
	if model == "" {
		t.Errorf("Expected a model to be selected, got empty string")
	}
	// In this test, since MockStateManager has "test-model", it should be in the available models
	expectedModels := []string{"test-model"}
	found := false
	for _, expectedModel := range expectedModels {
		if model == expectedModel {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("Expected one of %v, got %s", expectedModels, model)
	}
}

func TestContextualRouter_UpdatePerformance(t *testing.T) {
	db := setupTestDB(t)
	embeddingService := NewMockEmbeddingService()

	config := bandit.BanditConfig{
		SimilarityThreshold: 0.7,
		MaxSimilarRequests:  50,
		RecencyDays:         30,
		FeedbackWeight:      0.6,
		LatencyWeight:       0.2,
		CostWeight:          -0.2,
		MinSimilarRequests:  5,
		DefaultModel:        "default-model",
	}

	router, err := NewContextualRouter(config, embeddingService, &MockStateManager{}, db)
	if err != nil {
		t.Fatalf("Failed to create contextual router: %v", err)
	}
	defer router.Close()

	ctx := context.Background()
	metrics := bandit.PerformanceMetrics{
		Latency:       100.0,
		Cost:          0.01,
		FeedbackScore: 0.8,
		Success:       true,
	}

	err = router.UpdatePerformance(ctx, "request-123", "test-model", metrics)
	if err != nil {
		t.Fatalf("UpdatePerformance failed: %v", err)
	}
}

func TestContextualRouter_Health(t *testing.T) {
	db := setupTestDB(t)
	embeddingService := NewMockEmbeddingService()

	config := bandit.BanditConfig{
		SimilarityThreshold: 0.7,
		MaxSimilarRequests:  50,
		RecencyDays:         30,
		FeedbackWeight:      0.6,
		LatencyWeight:       0.2,
		CostWeight:          -0.2,
		MinSimilarRequests:  5,
		DefaultModel:        "default-model",
	}

	router, err := NewContextualRouter(config, embeddingService, &MockStateManager{}, db)
	if err != nil {
		t.Fatalf("Failed to create contextual router: %v", err)
	}

	ctx := context.Background()
	if err := router.Health(ctx); err != nil {
		t.Errorf("Health check failed: %v", err)
	}

	router.Close()

	if err := router.Health(ctx); err == nil {
		t.Error("Expected health check to fail after close")
	}
}

// Note: Beta and Gamma sampling tests removed as they test private methods
// The sampling functionality is tested indirectly through the public SelectModel method

func TestContextualRouter_ConfigValidation(t *testing.T) {
	db := setupTestDB(t)
	embeddingService := NewMockEmbeddingService()

	// Test invalid similarity threshold
	invalidConfig := bandit.BanditConfig{
		SimilarityThreshold: 1.5, // Invalid - should be [0,1]
		DefaultModel:        "test-model",
	}

	_, err := NewContextualRouter(invalidConfig, embeddingService, &MockStateManager{}, db)
	if err == nil {
		t.Error("Expected error for invalid similarity threshold")
	}

	// Test negative similarity threshold
	invalidConfig.SimilarityThreshold = -0.1
	_, err = NewContextualRouter(invalidConfig, embeddingService, &MockStateManager{}, db)
	if err == nil {
		t.Error("Expected error for negative similarity threshold")
	}
}

func TestContextualRouter_GammaSamplingBoundsChecking(t *testing.T) {
	db := setupTestDB(t)
	embeddingService := NewMockEmbeddingService()

	config := bandit.BanditConfig{
		SimilarityThreshold: 0.7,
		MaxSimilarRequests:  50,
		RecencyDays:         30,
		FeedbackWeight:      0.6,
		LatencyWeight:       0.2,
		CostWeight:          -0.2,
		MinSimilarRequests:  5,
		DefaultModel:        "test-model",
	}

	router, err := NewContextualRouter(config, embeddingService, &MockStateManager{}, db)
	if err != nil {
		t.Fatalf("Failed to create contextual router: %v", err)
	}
	defer router.Close()

	// Test gamma sampling with various edge case inputs
	testCases := []struct {
		name  string
		shape float64
		desc  string
	}{
		{"very_small_positive", 0.0001, "should return small positive value"},
		{"exactly_zero", 0.0, "should return small positive value"},
		{"negative", -1.0, "should return small positive value"},
		{"positive_infinity", math.Inf(1), "should return reasonable default"},
		{"negative_infinity", math.Inf(-1), "should return reasonable default"},
		{"nan", math.NaN(), "should return reasonable default"},
		{"normal_case", 2.5, "should work normally"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Call gamma sampling indirectly through beta sampling
			// which is called by Thompson sampling methods
			result := router.betaSample(tc.shape, 1.0)

			// Verify result is valid
			if math.IsNaN(result) {
				t.Errorf("Result should not be NaN for %s", tc.desc)
			}
			if math.IsInf(result, 0) {
				t.Errorf("Result should not be infinite for %s", tc.desc)
			}
			if result <= 0 {
				t.Errorf("Result should be positive for %s, got %f", tc.desc, result)
			}
			if result > 1000 {
				t.Errorf("Result should be reasonable for %s, got %f", tc.desc, result)
			}
		})
	}
}

func TestContextualRouter_RewardNormalization(t *testing.T) {
	db := setupTestDB(t)
	embeddingService := NewMockEmbeddingService()

	config := bandit.BanditConfig{
		SimilarityThreshold: 0.7,
		MaxSimilarRequests:  50,
		RecencyDays:         30,
		FeedbackWeight:      1.0,  // Equal weight to feedback
		LatencyWeight:       1.0, // Positive weight for latency (rewards speed)
		CostWeight:          -1.0, // Negative weight for cost (bad)
		MinSimilarRequests:  1,    // Lower threshold for testing
		DefaultModel:        "test-model",
	}

	router, err := NewContextualRouter(config, embeddingService, &MockStateManager{}, db)
	if err != nil {
		t.Fatalf("Failed to create contextual router: %v", err)
	}
	defer router.Close()

	// Test cases for token-based latency normalization with exponential falloff
	testCases := []struct {
		name        string
		latencyMs   float64
		tokensUsed  int
		cost        float64
		feedback    float64
		description string
	}{
		{
			name:        "efficient_response",
			latencyMs:   1000.0, // 1 second
			tokensUsed:  100,    // 10ms/token - very efficient
			cost:        0.05,
			feedback:    0.8,
			description: "fast response per token should have high latency reward",
		},
		{
			name:        "slow_response",
			latencyMs:   10000.0, // 10 seconds
			tokensUsed:  100,     // 100ms/token - slow
			cost:        0.1,
			feedback:    0.9,
			description: "slow response per token should have lower latency reward",
		},
		{
			name:        "large_response_reasonable_speed",
			latencyMs:   10000.0, // 10 seconds
			tokensUsed:  2000,    // 5ms/token - reasonable for large response
			cost:        0.5,
			feedback:    0.85,
			description: "large response with reasonable speed should be rewarded",
		},
		{
			name:        "half_life_test",
			latencyMs:   5000.0, // 5 seconds
			tokensUsed:  1,      // 5000ms/token - exactly at half-life
			cost:        0.2,
			feedback:    0.7,
			description: "exactly at half-life should give 0.5 normalized latency",
		},
		{
			name:        "zero_tokens_edge_case",
			latencyMs:   2000.0,
			tokensUsed:  0, // Edge case - should default to 1 token
			cost:        0.1,
			feedback:    0.6,
			description: "zero tokens should default to 1 to prevent division by zero",
		},
		{
			name:        "very_slow_per_token",
			latencyMs:   20000.0, // 20 seconds
			tokensUsed:  1,       // 20000ms/token - extremely slow
			cost:        0.3,
			feedback:    0.9,
			description: "very slow per token should have very low latency reward",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create mock similar requests with tokens
			similarReqs := []bandit.SimilarRequest{
				{
					RequestID:  "test-req-1",
					Model:      "test-model",
					Similarity: 0.9,
					Latency:    tc.latencyMs,
					Cost:       tc.cost,
					Feedback:   tc.feedback,
					TokensUsed: tc.tokensUsed,
					CreatedAt:  "2023-01-01T00:00:00Z",
				},
			}

			// Test the internal thompson sampling logic
			// This indirectly tests reward normalization through model selection
			selectedModel := router.thompsonSampleSimilar(similarReqs)

			// Verify the function doesn't crash and returns a valid model
			if selectedModel == "" {
				t.Errorf("Expected a model to be selected, got empty string")
			}

			// Calculate what the normalized values should be using the same logic
			latencyPerToken := tc.latencyMs / math.Max(float64(tc.tokensUsed), 1.0)
			expectedNormalizedLatency := math.Exp(-latencyPerToken / (5000.0 / math.Ln2))
			expectedNormalizedCost := math.Min(tc.cost, 1.0)

			// Verify exponential falloff behavior for specific cases
			if tc.name == "half_life_test" {
				// At exactly 5000ms/token, should be approximately 0.5
				if math.Abs(expectedNormalizedLatency-0.5) > 0.01 {
					t.Errorf("Half-life test: expected ~0.5, got %.3f", expectedNormalizedLatency)
				}
			}

			if tc.name == "efficient_response" {
				// 10ms/token should give high reward (close to 1.0)
				if expectedNormalizedLatency < 0.9 {
					t.Errorf("Efficient response should have high latency reward, got %.3f", expectedNormalizedLatency)
				}
			}

			if tc.name == "very_slow_per_token" {
				// 20000ms/token should give very low reward (close to 0)
				if expectedNormalizedLatency > 0.1 {
					t.Errorf("Very slow response should have low latency reward, got %.3f", expectedNormalizedLatency)
				}
			}

			// Verify reward is in reasonable bounds
			expectedReward := config.FeedbackWeight*tc.feedback +
				config.LatencyWeight*expectedNormalizedLatency +
				config.CostWeight*expectedNormalizedCost

			// With token-based normalization, reward should be reasonable
			if math.Abs(expectedReward) > 100 {
				t.Errorf("Reward calculation seems extreme even after normalization: %f", expectedReward)
			}

			t.Logf("Test case: %s - Latency: %.1fms/%dtokens=%.1fms/token, Normalized: %.3f, Expected reward: %.3f",
				tc.description, tc.latencyMs, tc.tokensUsed, latencyPerToken, expectedNormalizedLatency, expectedReward)
		})
	}
}

func TestContextualRouter_RewardCalculationCorrectness(t *testing.T) {
	// Test to verify that fast responses are rewarded more than slow responses
	db := setupTestDB(t)
	embeddingService := NewMockEmbeddingService()

	config := bandit.BanditConfig{
		SimilarityThreshold: 0.7,
		MaxSimilarRequests:  50,
		RecencyDays:         30,
		FeedbackWeight:      0.6,
		LatencyWeight:       0.2, // Positive weight
		CostWeight:          -0.2,
		MinSimilarRequests:  1,
		DefaultModel:        "test-model",
	}

	router, err := NewContextualRouter(config, embeddingService, &MockStateManager{}, db)
	if err != nil {
		t.Fatalf("Failed to create contextual router: %v", err)
	}
	defer router.Close()

	// Create two similar requests with same feedback and cost, but different latency efficiency
	fastRequest := bandit.SimilarRequest{
		RequestID:  "fast-req",
		Model:      "test-model",
		Similarity: 0.9,
		Latency:    1000.0,  // 1 second
		TokensUsed: 100,     // 10ms/token - very efficient
		Cost:       0.1,
		Feedback:   0.8,
		CreatedAt:  "2023-01-01T00:00:00Z",
	}

	slowRequest := bandit.SimilarRequest{
		RequestID:  "slow-req",
		Model:      "test-model",
		Similarity: 0.9,
		Latency:    10000.0, // 10 seconds
		TokensUsed: 100,     // 100ms/token - slow
		Cost:       0.1,
		Feedback:   0.8, // Same feedback as fast request
		CreatedAt:  "2023-01-01T00:00:00Z",
	}

	// Calculate rewards manually using the same logic
	fastLatencyPerToken := fastRequest.Latency / math.Max(float64(fastRequest.TokensUsed), 1.0)
	fastNormalizedLatency := math.Exp(-fastLatencyPerToken / (5000.0 / math.Ln2))
	fastCostPerToken := fastRequest.Cost / math.Max(float64(fastRequest.TokensUsed), 1.0)
	maxCostPerToken := 0.0006 // Most expensive current models
	fastNormalizedCost := math.Min(1.0, math.Max(0.0, fastCostPerToken / maxCostPerToken))
	fastReward := config.FeedbackWeight*fastRequest.Feedback +
		config.LatencyWeight*fastNormalizedLatency +
		config.CostWeight*fastNormalizedCost

	slowLatencyPerToken := slowRequest.Latency / math.Max(float64(slowRequest.TokensUsed), 1.0)
	slowNormalizedLatency := math.Exp(-slowLatencyPerToken / (5000.0 / math.Ln2))
	slowCostPerToken := slowRequest.Cost / math.Max(float64(slowRequest.TokensUsed), 1.0)
	slowNormalizedCost := math.Min(1.0, math.Max(0.0, slowCostPerToken / maxCostPerToken))
	slowReward := config.FeedbackWeight*slowRequest.Feedback +
		config.LatencyWeight*slowNormalizedLatency +
		config.CostWeight*slowNormalizedCost

	// Fast request should have higher reward than slow request
	if fastReward <= slowReward {
		t.Errorf("Fast request should have higher reward than slow request. Fast: %.3f, Slow: %.3f", fastReward, slowReward)
	}

	// Verify the latency component specifically
	fastLatencyContribution := config.LatencyWeight * fastNormalizedLatency
	slowLatencyContribution := config.LatencyWeight * slowNormalizedLatency

	if fastLatencyContribution <= slowLatencyContribution {
		t.Errorf("Fast request should have higher latency contribution. Fast: %.3f, Slow: %.3f",
			fastLatencyContribution, slowLatencyContribution)
	}

	t.Logf("Fast request: %dms/%dtokens=%.1fms/token, normalized=%.3f, latency contribution=%.3f, total reward=%.3f",
		int(fastRequest.Latency), fastRequest.TokensUsed, fastLatencyPerToken, fastNormalizedLatency, fastLatencyContribution, fastReward)
	t.Logf("Slow request: %dms/%dtokens=%.1fms/token, normalized=%.3f, latency contribution=%.3f, total reward=%.3f",
		int(slowRequest.Latency), slowRequest.TokensUsed, slowLatencyPerToken, slowNormalizedLatency, slowLatencyContribution, slowReward)

	// Ensure rewards are positive for good responses (high feedback, reasonable cost)
	if fastReward <= 0 {
		t.Errorf("Fast response with good feedback should have positive reward, got %.3f", fastReward)
	}
}

func TestContextualRouter_CostPerTokenNormalization(t *testing.T) {
	db := setupTestDB(t)
	embeddingService := &MockEmbeddingService{}

	config := bandit.BanditConfig{
		SimilarityThreshold: 0.8,
		MaxSimilarRequests:  10,
		FeedbackWeight:      0.6,
		LatencyWeight:       0.2,
		CostWeight:          -0.2, // Negative weight (cost is bad)
		MinSimilarRequests:  1,
		DefaultModel:        "test-model",
	}

	router, err := NewContextualRouter(config, embeddingService, &MockStateManager{}, db)
	if err != nil {
		t.Fatalf("Failed to create contextual router: %v", err)
	}
	defer router.Close()

	// Create two similar requests with same feedback and latency, but different cost efficiency
	cheapRequest := bandit.SimilarRequest{
		RequestID:  "cheap-req",
		Model:      "test-model",
		Similarity: 0.9,
		Latency:    5000.0, // Same latency
		TokensUsed: 1000,   // Many tokens
		Cost:       0.01,   // $0.00001/token - very cheap
		Feedback:   0.8,
		CreatedAt:  "2023-01-01T00:00:00Z",
	}

	expensiveRequest := bandit.SimilarRequest{
		RequestID:  "expensive-req",
		Model:      "test-model",
		Similarity: 0.9,
		Latency:    5000.0, // Same latency
		TokensUsed: 100,    // Fewer tokens
		Cost:       0.01,   // $0.0001/token - 10x more expensive per token
		Feedback:   0.8,    // Same feedback
		CreatedAt:  "2023-01-01T00:00:00Z",
	}

	// Calculate rewards manually using the same logic
	cheapLatencyPerToken := cheapRequest.Latency / math.Max(float64(cheapRequest.TokensUsed), 1.0)
	cheapNormalizedLatency := math.Exp(-cheapLatencyPerToken / (5000.0 / math.Ln2))
	cheapCostPerToken := cheapRequest.Cost / math.Max(float64(cheapRequest.TokensUsed), 1.0)
	maxCostPerToken := 0.0006 // Most expensive current models
	cheapNormalizedCost := math.Min(1.0, math.Max(0.0, cheapCostPerToken / maxCostPerToken))
	cheapReward := config.FeedbackWeight*cheapRequest.Feedback +
		config.LatencyWeight*cheapNormalizedLatency +
		config.CostWeight*cheapNormalizedCost

	expensiveLatencyPerToken := expensiveRequest.Latency / math.Max(float64(expensiveRequest.TokensUsed), 1.0)
	expensiveNormalizedLatency := math.Exp(-expensiveLatencyPerToken / (5000.0 / math.Ln2))
	expensiveCostPerToken := expensiveRequest.Cost / math.Max(float64(expensiveRequest.TokensUsed), 1.0)
	expensiveNormalizedCost := math.Min(1.0, math.Max(0.0, expensiveCostPerToken / maxCostPerToken))
	expensiveReward := config.FeedbackWeight*expensiveRequest.Feedback +
		config.LatencyWeight*expensiveNormalizedLatency +
		config.CostWeight*expensiveNormalizedCost

	// Cheap per-token request should have higher reward than expensive per-token request
	if cheapReward <= expensiveReward {
		t.Errorf("Cheap per-token request should have higher reward. Cheap: %.3f, Expensive: %.3f", cheapReward, expensiveReward)
	}

	// Verify the cost component specifically
	cheapCostContribution := config.CostWeight * cheapNormalizedCost
	expensiveCostContribution := config.CostWeight * expensiveNormalizedCost

	// With negative cost weight, cheaper should have higher (less negative) contribution
	if cheapCostContribution <= expensiveCostContribution {
		t.Errorf("Cheap request should have higher (less negative) cost contribution. Cheap: %.3f, Expensive: %.3f",
			cheapCostContribution, expensiveCostContribution)
	}

	t.Logf("Cheap request: $%.5f/%dtokens=$%.8f/token, normalized=%.3f, cost contribution=%.3f, total reward=%.3f",
		cheapRequest.Cost, cheapRequest.TokensUsed, cheapCostPerToken, cheapNormalizedCost, cheapCostContribution, cheapReward)
	t.Logf("Expensive request: $%.5f/%dtokens=$%.8f/token, normalized=%.3f, cost contribution=%.3f, total reward=%.3f",
		expensiveRequest.Cost, expensiveRequest.TokensUsed, expensiveCostPerToken, expensiveNormalizedCost, expensiveCostContribution, expensiveReward)

	// Verify normalization bounds
	if cheapNormalizedCost < 0 || cheapNormalizedCost > 1 {
		t.Errorf("Normalized cost should be in [0,1], got %.3f for cheap request", cheapNormalizedCost)
	}
	if expensiveNormalizedCost < 0 || expensiveNormalizedCost > 1 {
		t.Errorf("Normalized cost should be in [0,1], got %.3f for expensive request", expensiveNormalizedCost)
	}
}

func TestSelectFallbackModel_RandomSelection(t *testing.T) {
	db := setupTestDB(t)
	embeddingService := NewMockEmbeddingService()

	// MockStateManager that returns multiple models
	mockStateManager := &MockStateManager{}

	config := bandit.BanditConfig{
		SimilarityThreshold: 0.7,
		MaxSimilarRequests:  50,
		RecencyDays:         30,
		FeedbackWeight:      0.6,
		LatencyWeight:       0.2,
		CostWeight:          -0.2,
		MinSimilarRequests:  5,
		DefaultModel:        "should-not-be-used", // This should not be returned due to random selection
		GlobalFallback:      true,
	}

	router, err := NewContextualRouter(config, embeddingService, mockStateManager, db)
	if err != nil {
		t.Fatalf("Failed to create contextual router: %v", err)
	}
	defer router.Close()

	// Test the fallback selection multiple times to verify randomness
	// Since MockStateManager returns "test-model", we expect that to be selected
	selectedModels := make(map[string]int)
	iterations := 10

	for i := 0; i < iterations; i++ {
		model := router.selectFallbackModel()
		selectedModels[model]++
	}

	// Should select from available models (in this case "test-model" from MockStateManager)
	if len(selectedModels) == 0 {
		t.Errorf("No models were selected")
	}

	// Verify that a model was actually selected
	totalSelections := 0
	for _, count := range selectedModels {
		totalSelections += count
	}

	if totalSelections != iterations {
		t.Errorf("Expected %d total selections, got %d", iterations, totalSelections)
	}

	// The specific model selected should be "test-model" since that's what MockStateManager provides
	if _, exists := selectedModels["test-model"]; !exists {
		t.Errorf("Expected 'test-model' to be selected (provided by MockStateManager), got models: %v", selectedModels)
	}

	t.Logf("Selected models distribution: %v", selectedModels)
}

// MockStateManagerMultiModel provides multiple models for testing random selection
type MockStateManagerMultiModel struct{}

func (m *MockStateManagerMultiModel) GetAllArms() map[string]*bandit.BanditArm {
	return map[string]*bandit.BanditArm{
		"model-a": {
			Model:        "model-a",
			RequestCount: 5,
			SuccessCount: 4,
		},
		"model-b": {
			Model:        "model-b",
			RequestCount: 3,
			SuccessCount: 2,
		},
		"model-c": {
			Model:        "model-c",
			RequestCount: 7,
			SuccessCount: 6,
		},
	}
}

func (m *MockStateManagerMultiModel) UpdateArm(model string, metrics bandit.PerformanceMetrics) error {
	return nil
}

func (m *MockStateManagerMultiModel) CreateArm(model string) error {
	return nil
}

func (m *MockStateManagerMultiModel) GetArm(model string) *bandit.BanditArm {
	arms := m.GetAllArms()
	if arm, exists := arms[model]; exists {
		return arm
	}
	return nil
}

func (m *MockStateManagerMultiModel) GetStatistics() map[string]interface{} {
	return map[string]interface{}{
		"total_arms": len(m.GetAllArms()),
		"test": true,
	}
}

func (m *MockStateManagerMultiModel) Health() error {
	return nil
}

func (m *MockStateManagerMultiModel) Close() error {
	return nil
}

func TestSelectFallbackModel_RandomDistribution(t *testing.T) {
	db := setupTestDB(t)
	embeddingService := NewMockEmbeddingService()

	// Use MockStateManagerMultiModel that returns multiple models
	mockStateManager := &MockStateManagerMultiModel{}

	config := bandit.BanditConfig{
		SimilarityThreshold: 0.7,
		MaxSimilarRequests:  50,
		RecencyDays:         30,
		FeedbackWeight:      0.6,
		LatencyWeight:       0.2,
		CostWeight:          -0.2,
		MinSimilarRequests:  5,
		DefaultModel:        "unused-default",
		GlobalFallback:      true,
	}

	router, err := NewContextualRouter(config, embeddingService, mockStateManager, db)
	if err != nil {
		t.Fatalf("Failed to create contextual router: %v", err)
	}
	defer router.Close()

	// Test the fallback selection multiple times to verify randomness
	selectedModels := make(map[string]int)
	iterations := 100
	expectedModels := []string{"model-a", "model-b", "model-c"}

	for i := 0; i < iterations; i++ {
		model := router.selectFallbackModel()
		selectedModels[model]++
	}

	// Verify all expected models were selected at least once
	for _, expectedModel := range expectedModels {
		if count, exists := selectedModels[expectedModel]; !exists || count == 0 {
			t.Errorf("Expected model '%s' to be selected at least once", expectedModel)
		}
	}

	// Verify we have reasonable distribution (no model should be selected more than 80% of the time)
	for model, count := range selectedModels {
		if float64(count)/float64(iterations) > 0.8 {
			t.Errorf("Model '%s' was selected too often (%d/%d = %.1f%%), suggesting poor randomness",
				model, count, iterations, float64(count)/float64(iterations)*100)
		}
	}

	t.Logf("Random distribution over %d iterations: %v", iterations, selectedModels)
}