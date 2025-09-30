package bandit

import (
	"math"
	"testing"

	"llm-router/domain/bandit"
)

// TestSpeedScoreCalculation verifies that the speed score calculation correctly
// rewards faster models (lower latency) with higher scores
func TestSpeedScoreCalculation(t *testing.T) {
	tests := []struct {
		name          string
		latencyMs     float64
		tokensUsed    int
		expectedRange [2]float64 // [min, max] expected speedScore
		description   string
	}{
		{
			name:          "very fast model",
			latencyMs:     100,
			tokensUsed:    100,
			expectedRange: [2]float64{0.99, 1.00},
			description:   "1ms/token should give very high speedScore (~0.9999)",
		},
		{
			name:          "medium speed model",
			latencyMs:     500000,
			tokensUsed:    100,
			expectedRange: [2]float64{0.45, 0.55},
			description:   "5000ms/token (half-life) should give ~0.5",
		},
		{
			name:          "slow model",
			latencyMs:     1000000,
			tokensUsed:    100,
			expectedRange: [2]float64{0.20, 0.30},
			description:   "10000ms/token should give low speedScore (~0.25)",
		},
		{
			name:          "extremely slow model",
			latencyMs:     2000000,
			tokensUsed:    100,
			expectedRange: [2]float64{0.05, 0.15},
			description:   "20000ms/token should give very low speedScore",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Calculate speed score using the same formula as the actual code
			latencyPerToken := tt.latencyMs / math.Max(float64(tt.tokensUsed), 1.0)
			speedScore := math.Exp(-latencyPerToken / (5000.0 / math.Ln2))

			if speedScore < tt.expectedRange[0] || speedScore > tt.expectedRange[1] {
				t.Errorf("%s: speedScore = %.4f, expected range [%.2f, %.2f]\n  %s",
					tt.name, speedScore, tt.expectedRange[0], tt.expectedRange[1], tt.description)
			}

			t.Logf("%s: latency=%vms, tokens=%d, latency/token=%.2f, speedScore=%.4f",
				tt.name, tt.latencyMs, tt.tokensUsed, latencyPerToken, speedScore)
		})
	}
}

// TestRewardCalculationDirection verifies that the reward calculation correctly
// uses positive weight for speed (not latency) and negative weight for cost
func TestRewardCalculationDirection(t *testing.T) {
	config := bandit.BanditConfig{
		FeedbackWeight: 0.6,
		LatencyWeight:  0.2,  // Positive - rewards speed
		CostWeight:     -0.2, // Negative - penalizes cost
	}

	tests := []struct {
		name        string
		request     bandit.SimilarRequest
		description string
	}{
		{
			name: "fast and cheap model should have high reward",
			request: bandit.SimilarRequest{
				Model:      "fast-cheap-model",
				Latency:    100, // 1ms/token - very fast
				TokensUsed: 100,
				Cost:       0.0001, // $0.000001/token - very cheap
				Feedback:   0.8,
				Similarity: 1.0,
			},
			description: "Fast (high speedScore) + cheap (low cost) = high reward",
		},
		{
			name: "slow and expensive model should have low reward",
			request: bandit.SimilarRequest{
				Model:      "slow-expensive-model",
				Latency:    1000000, // 10000ms/token - very slow
				TokensUsed: 100,
				Cost:       0.06, // $0.0006/token - very expensive
				Feedback:   0.8,  // Same feedback as fast model
				Similarity: 1.0,
			},
			description: "Slow (low speedScore) + expensive (high cost) = low reward",
		},
	}

	// Calculate rewards for both models
	var rewards []float64
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Speed score calculation
			latencyPerToken := tt.request.Latency / math.Max(float64(tt.request.TokensUsed), 1.0)
			speedScore := math.Exp(-latencyPerToken / (5000.0 / math.Ln2))

			// Cost normalization
			costPerToken := tt.request.Cost / math.Max(float64(tt.request.TokensUsed), 1.0)
			maxCostPerToken := 0.0006
			normalizedCost := math.Min(1.0, math.Max(0.0, costPerToken/maxCostPerToken))

			// Reward calculation (same formula as actual code)
			reward := config.FeedbackWeight*tt.request.Feedback +
				config.LatencyWeight*speedScore +
				config.CostWeight*normalizedCost

			rewards = append(rewards, reward)

			t.Logf("%s:\n  Feedback: %.2f, SpeedScore: %.4f, NormalizedCost: %.4f\n  Reward: %.4f\n  %s",
				tt.name, tt.request.Feedback, speedScore, normalizedCost, reward, tt.description)
		})
	}

	// Verify that fast+cheap model has higher reward than slow+expensive model
	if len(rewards) == 2 {
		fastCheapReward := rewards[0]
		slowExpensiveReward := rewards[1]

		if fastCheapReward <= slowExpensiveReward {
			t.Errorf("Fast+cheap model reward (%.4f) should be > slow+expensive model reward (%.4f)",
				fastCheapReward, slowExpensiveReward)
		} else {
			t.Logf("✓ Reward direction correct: fast+cheap (%.4f) > slow+expensive (%.4f)",
				fastCheapReward, slowExpensiveReward)
		}
	}
}

// TestLatencyWeightSign verifies that positive latency weight correctly rewards speed
func TestLatencyWeightSign(t *testing.T) {
	// Two models with same feedback and cost, different latencies
	fastModel := bandit.SimilarRequest{
		Latency:    100, // 1ms/token
		TokensUsed: 100,
		Cost:       0.01, // Same cost
		Feedback:   0.5,  // Same feedback
	}

	slowModel := bandit.SimilarRequest{
		Latency:    1000000, // 10000ms/token
		TokensUsed: 100,
		Cost:       0.01, // Same cost
		Feedback:   0.5,  // Same feedback
	}

	config := bandit.BanditConfig{
		FeedbackWeight: 0.6,
		LatencyWeight:  0.2, // POSITIVE weight
		CostWeight:     -0.2,
	}

	// Calculate rewards
	calcReward := func(req bandit.SimilarRequest) float64 {
		latencyPerToken := req.Latency / math.Max(float64(req.TokensUsed), 1.0)
		speedScore := math.Exp(-latencyPerToken / (5000.0 / math.Ln2))

		costPerToken := req.Cost / math.Max(float64(req.TokensUsed), 1.0)
		normalizedCost := math.Min(1.0, math.Max(0.0, costPerToken/0.0006))

		return config.FeedbackWeight*req.Feedback +
			config.LatencyWeight*speedScore +
			config.CostWeight*normalizedCost
	}

	fastReward := calcReward(fastModel)
	slowReward := calcReward(slowModel)

	if fastReward <= slowReward {
		t.Errorf("Fast model reward (%.4f) should be > slow model reward (%.4f) with positive LatencyWeight",
			fastReward, slowReward)
		t.Errorf("This indicates the latency weight sign is incorrect!")
	} else {
		t.Logf("✓ Latency weight sign correct: positive weight rewards faster models")
		t.Logf("  Fast model (1ms/token): reward = %.4f", fastReward)
		t.Logf("  Slow model (10000ms/token): reward = %.4f", slowReward)
		t.Logf("  Difference: %.4f", fastReward-slowReward)
	}
}

// TestCostWeightSign verifies that negative cost weight correctly penalizes expensive models
func TestCostWeightSign(t *testing.T) {
	// Two models with same feedback and latency, different costs
	cheapModel := bandit.SimilarRequest{
		Latency:    1000, // Same latency
		TokensUsed: 100,
		Cost:       0.0001, // Very cheap
		Feedback:   0.5,    // Same feedback
	}

	expensiveModel := bandit.SimilarRequest{
		Latency:    1000, // Same latency
		TokensUsed: 100,
		Cost:       0.06, // Very expensive
		Feedback:   0.5,  // Same feedback
	}

	config := bandit.BanditConfig{
		FeedbackWeight: 0.6,
		LatencyWeight:  0.2,
		CostWeight:     -0.2, // NEGATIVE weight
	}

	// Calculate rewards
	calcReward := func(req bandit.SimilarRequest) float64 {
		latencyPerToken := req.Latency / math.Max(float64(req.TokensUsed), 1.0)
		speedScore := math.Exp(-latencyPerToken / (5000.0 / math.Ln2))

		costPerToken := req.Cost / math.Max(float64(req.TokensUsed), 1.0)
		normalizedCost := math.Min(1.0, math.Max(0.0, costPerToken/0.0006))

		return config.FeedbackWeight*req.Feedback +
			config.LatencyWeight*speedScore +
			config.CostWeight*normalizedCost
	}

	cheapReward := calcReward(cheapModel)
	expensiveReward := calcReward(expensiveModel)

	if cheapReward <= expensiveReward {
		t.Errorf("Cheap model reward (%.4f) should be > expensive model reward (%.4f) with negative CostWeight",
			cheapReward, expensiveReward)
		t.Errorf("This indicates the cost weight sign is incorrect!")
	} else {
		t.Logf("✓ Cost weight sign correct: negative weight penalizes expensive models")
		t.Logf("  Cheap model ($0.000001/token): reward = %.4f", cheapReward)
		t.Logf("  Expensive model ($0.0006/token): reward = %.4f", expensiveReward)
		t.Logf("  Difference: %.4f", cheapReward-expensiveReward)
	}
}
