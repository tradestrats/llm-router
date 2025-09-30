package bandit

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"

	"llm-router/domain/bandit"
	"llm-router/domain/chat"
	"llm-router/domain/embedding"

	"github.com/pgvector/pgvector-go"
	"github.com/sirupsen/logrus"
	"gorm.io/gorm"
)

// ContextualRouter implements bandit.SimilarityRouter using embeddings and Thompson sampling
type ContextualRouter struct {
	config           bandit.BanditConfig
	embeddingService embedding.EmbeddingService
	stateManager     bandit.StateManager
	db               *gorm.DB
	rng              *rand.Rand
	mu               sync.RWMutex
	globalArms       map[string]*bandit.BanditArm
	closed           bool
}

// NewContextualRouter creates a new contextual bandit router
func NewContextualRouter(
	config bandit.BanditConfig,
	embeddingService embedding.EmbeddingService,
	stateManager bandit.StateManager,
	db *gorm.DB,
) (*ContextualRouter, error) {
	// Validate and set defaults for config
	if config.SimilarityThreshold < 0 || config.SimilarityThreshold > 1 {
		return nil, fmt.Errorf("similarity threshold must be between 0 and 1")
	}
	if config.MaxSimilarRequests <= 0 {
		config.MaxSimilarRequests = 50
	}
	if config.RecencyDays <= 0 {
		config.RecencyDays = 30
	}
	if config.MinSimilarRequests <= 0 {
		config.MinSimilarRequests = 5
	}
	if config.MinConfidenceScore <= 0 {
		config.MinConfidenceScore = 0.1
	}
	if config.OptimisticPrior <= 0 {
		config.OptimisticPrior = 0.8 // Assume 80% success rate for new models
	}
	if config.ExplorationBonus <= 0 {
		config.ExplorationBonus = 0.1
	}
	if config.MinRequestsForGlobal <= 0 {
		config.MinRequestsForGlobal = 10
	}

	router := &ContextualRouter{
		config:           config,
		embeddingService: embeddingService,
		stateManager:     stateManager,
		db:               db,
		rng:              rand.New(rand.NewSource(time.Now().UnixNano())),
		globalArms:       make(map[string]*bandit.BanditArm),
	}

	// Initialize global arms from database
	if err := router.loadGlobalArms(); err != nil {
		logrus.WithError(err).Warn("Failed to load global arms, continuing with empty state")
	}

	return router, nil
}

// SelectModel selects the best model using resilient decision hierarchy
func (cr *ContextualRouter) SelectModel(ctx context.Context, req *chat.Request) (string, error) {
	if err := cr.checkHealth(); err != nil {
		logrus.WithError(err).Debug("Health check failed, using fallback")
		return cr.selectFallbackModel(), nil
	}

	logrus.WithField("messages", len(req.Messages)).Debug("Starting resilient model selection")

	// Decision Hierarchy:
	// 1. Try contextual bandit with embedding similarity
	// 2. Fall back to global bandit arms (from state manager)
	// 3. Fall back to optimistic exploration of new models
	// 4. Ultimate fallback to default model

	// Step 1: Try contextual bandit selection
	if model, confidence, err := cr.tryContextualSelection(ctx, req); err == nil {
		if confidence >= cr.config.MinConfidenceScore {
			logrus.WithFields(logrus.Fields{
				"selected_model": model,
				"confidence":     confidence,
				"method":         "contextual",
			}).Info("Selected model using contextual bandit")
			return model, nil
		}
		logrus.WithFields(logrus.Fields{
			"model":      model,
			"confidence": confidence,
			"threshold":  cr.config.MinConfidenceScore,
		}).Debug("Contextual selection confidence too low, trying global bandit")
	} else {
		logrus.WithError(err).Debug("Contextual selection failed, trying global bandit")
	}

	// Step 2: Try global bandit selection
	if model, confidence, err := cr.tryGlobalBanditSelection(); err == nil {
		if confidence >= cr.config.MinConfidenceScore {
			logrus.WithFields(logrus.Fields{
				"selected_model": model,
				"confidence":     confidence,
				"method":         "global_bandit",
			}).Info("Selected model using global bandit")
			return model, nil
		}
		logrus.WithFields(logrus.Fields{
			"model":      model,
			"confidence": confidence,
			"threshold":  cr.config.MinConfidenceScore,
		}).Debug("Global bandit confidence too low, trying optimistic exploration")
	} else {
		logrus.WithError(err).Debug("Global bandit selection failed, trying optimistic exploration")
	}

	// Step 3: Try optimistic exploration for cold start
	if model := cr.selectOptimisticExploration(); model != "" {
		logrus.WithFields(logrus.Fields{
			"selected_model": model,
			"method":         "optimistic_exploration",
		}).Info("Selected model using optimistic exploration")
		return model, nil
	}

	// Step 4: Ultimate fallback
	logrus.WithFields(logrus.Fields{
		"selected_model": cr.config.DefaultModel,
		"method":         "fallback",
	}).Warn("Using fallback model selection")
	return cr.selectFallbackModel(), nil
}

// findSimilarRequests uses pgvector to find similar requests
func (cr *ContextualRouter) findSimilarRequests(ctx context.Context, queryEmbedding []float32) ([]bandit.SimilarRequest, error) {
	var results []struct {
		RequestID  string    `json:"request_id"`
		Model      string    `json:"model"`
		Similarity float64   `json:"similarity"`
		Latency    float64   `json:"latency"`
		Cost       float64   `json:"cost"`
		Feedback   float64   `json:"feedback"`
		TokensUsed int       `json:"tokens_used"`
		CreatedAt  time.Time `json:"created_at"`
	}

	// Convert embedding to pgVector format
	embeddingVector := pgvector.NewVector(queryEmbedding)

	// Query similar requests with real feedback data and sensible defaults
	// Feedback imputation strategy:
	// - Use real feedback score if available (f.score > 0)
	// - Default to 0.5 (neutral) for requests without feedback
	// - This allows Thompson sampling to work with partial feedback data
	err := cr.db.WithContext(ctx).Raw(`
		SELECT
			r.id as request_id,
			r.model,
			1 - (r.embedding <=> $1) as similarity,
			COALESCE(m.latency_ms, 0) as latency,
			COALESCE(m.total_cost, 0) as cost,
			COALESCE(f.score, 0.5) as feedback,  -- Real feedback or neutral default
			COALESCE(m.tokens_used, 0) as tokens_used,
			r.created_at
		FROM requests r
		LEFT JOIN request_metrics m ON r.id = m.request_id
		LEFT JOIN request_feedback f ON r.id = f.request_id AND f.score > 0
		WHERE r.embedding IS NOT NULL
			AND r.status = 'completed'
			AND r.created_at >= NOW() - INTERVAL $2
			AND (1 - (r.embedding <=> $1)) >= $3
		ORDER BY similarity DESC
		LIMIT $4`,
		embeddingVector,
		fmt.Sprintf("%d days", cr.config.RecencyDays),
		cr.config.SimilarityThreshold,
		cr.config.MaxSimilarRequests,
	).Scan(&results).Error

	if err != nil {
		return nil, fmt.Errorf("failed to query similar requests: %w", err)
	}

	// Convert to domain types and track feedback imputation
	similarRequests := make([]bandit.SimilarRequest, len(results))
	realFeedbackCount := 0
	for i, result := range results {
		similarRequests[i] = bandit.SimilarRequest{
			RequestID:  result.RequestID,
			Model:      result.Model,
			Similarity: result.Similarity,
			Latency:    result.Latency,
			Cost:       result.Cost,
			Feedback:   result.Feedback,
			TokensUsed: result.TokensUsed,
			CreatedAt:  result.CreatedAt.Format(time.RFC3339),
		}

		// Count requests with real feedback (not the default 0.5)
		if result.Feedback != 0.5 {
			realFeedbackCount++
		}
	}

	logrus.WithFields(logrus.Fields{
		"total_requests":    len(similarRequests),
		"real_feedback":     realFeedbackCount,
		"imputed_feedback":  len(similarRequests) - realFeedbackCount,
		"feedback_coverage": float64(realFeedbackCount) / float64(len(similarRequests)),
	}).Debug("Retrieved similar requests with feedback imputation")

	return similarRequests, nil
}

// thompsonSampleSimilar performs Thompson sampling on similar requests
func (cr *ContextualRouter) thompsonSampleSimilar(similarRequests []bandit.SimilarRequest) string {
	// Group similar requests by model
	modelGroups := make(map[string][]bandit.SimilarRequest)
	for _, req := range similarRequests {
		modelGroups[req.Model] = append(modelGroups[req.Model], req)
	}

	if len(modelGroups) == 0 {
		return cr.config.DefaultModel
	}

	// Calculate reward for each model and sample from Beta distribution
	type modelScore struct {
		model string
		score float64
	}

	var scores []modelScore
	cr.mu.RLock()
	defer cr.mu.RUnlock()

	for model, requests := range modelGroups {
		if len(requests) == 0 {
			continue
		}

		// Calculate aggregate metrics for this model on similar requests
		var totalReward, totalWeight float64
		for _, req := range requests {
			// Calculate compound reward based on feedback, latency, and cost

			// Speed score: Convert latency to speed reward using exponential decay
			// Lower latency → Higher speedScore → Better reward
			// Examples: 1ms/token → 0.86, 5000ms/token → 0.5 (half-life), 10000ms/token → 0.25
			latencyPerToken := req.Latency / math.Max(float64(req.TokensUsed), 1.0) // ms/token
			speedScore := math.Exp(-latencyPerToken / (5000.0 / math.Ln2))          // Range: [0, 1], higher is faster

			// Normalize cost by tokens ($/token) with linear scaling based on current LLM market pricing
			costPerToken := req.Cost / math.Max(float64(req.TokensUsed), 1.0) // $/token

			// Linear scale based on typical LLM pricing ranges (as of 2024)
			// Free models: ~$0 per token
			// Cheap models (Llama 3.2, Gemma): ~$0.000002/token
			// Mid-tier models (GPT-4o-mini, Claude Haiku): ~$0.000015/token
			// Premium models (GPT-4o, Claude Sonnet): ~$0.00015/token
			// Top-tier models (o1-preview): ~$0.0006/token
			maxCostPerToken := 0.0006 // Most expensive current models
			normalizedCost := math.Min(1.0, math.Max(0.0, costPerToken/maxCostPerToken))

			reward := cr.config.FeedbackWeight*req.Feedback +
				cr.config.LatencyWeight*speedScore +
				cr.config.CostWeight*normalizedCost

			// Weight by similarity (more similar requests have higher influence)
			weight := req.Similarity
			totalReward += reward * weight
			totalWeight += weight
		}

		avgReward := totalReward / totalWeight
		count := float64(len(requests))

		// Use Beta distribution for Thompson sampling
		// Map reward to Beta parameters (assuming rewards are normalized to [0,1])
		alpha := math.Max(1, avgReward*count+1)
		beta := math.Max(1, (1-avgReward)*count+1)

		// Sample from Beta distribution
		sample := cr.betaSample(alpha, beta)
		scores = append(scores, modelScore{model: model, score: sample})
	}

	if len(scores) == 0 {
		return cr.config.DefaultModel
	}

	// Sort by score and select the highest
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	return scores[0].model
}

// selectWithGlobalBandit performs global Thompson sampling
func (cr *ContextualRouter) selectWithGlobalBandit() string {
	cr.mu.RLock()
	defer cr.mu.RUnlock()

	if len(cr.globalArms) == 0 {
		return cr.config.DefaultModel
	}

	type modelScore struct {
		model string
		score float64
	}

	var scores []modelScore
	for model, arm := range cr.globalArms {
		if arm.RequestCount == 0 {
			// Optimistic initialization for new arms
			scores = append(scores, modelScore{model: model, score: 1.0})
			continue
		}

		// Calculate success rate (used for Beta distribution parameters below)

		// Use Beta distribution
		alpha := math.Max(1, float64(arm.SuccessCount)+1)
		beta := math.Max(1, float64(arm.RequestCount-arm.SuccessCount)+1)

		sample := cr.betaSample(alpha, beta)
		scores = append(scores, modelScore{model: model, score: sample})
	}

	if len(scores) == 0 {
		return cr.config.DefaultModel
	}

	// Sort by score and select the highest
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	return scores[0].model
}

// betaSample samples from a Beta distribution
func (cr *ContextualRouter) betaSample(alpha, beta float64) float64 {
	// Simple beta sampling using gamma distributions
	x := cr.gammaSample(alpha)
	y := cr.gammaSample(beta)
	return x / (x + y)
}

// gammaSample samples from a Gamma distribution (simple approximation)
func (cr *ContextualRouter) gammaSample(shape float64) float64 {
	// Bounds checking to prevent infinite loops and invalid values
	if shape <= 0.001 {
		return 0.001 // Return small positive value for degenerate cases
	}
	if math.IsInf(shape, 0) || math.IsNaN(shape) {
		return 1.0 // Return reasonable default for invalid inputs
	}

	if shape < 1 {
		return cr.gammaSample(shape+1) * math.Pow(cr.rng.Float64(), 1.0/shape)
	}

	// Marsaglia and Tsang method for shape >= 1
	d := shape - 1.0/3.0
	c := 1.0 / math.Sqrt(9.0*d)

	for {
		v := -1.0
		for v <= 0 {
			x := cr.rng.NormFloat64()
			v = 1.0 + c*x
		}
		v = v * v * v
		x := (v - 1.0) / c
		u := cr.rng.Float64()
		if u < 1.0-0.0331*x*x*x*x || math.Log(u) < 0.5*x*x+d*(1.0-v+math.Log(v)) {
			result := d * v
			// Additional safety check on the result
			if math.IsInf(result, 0) || math.IsNaN(result) || result <= 0 {
				return 1.0 // Return reasonable default if calculation goes wrong
			}
			return result
		}
	}
}

// UpdatePerformance updates the bandit state with new performance data
func (cr *ContextualRouter) UpdatePerformance(ctx context.Context, requestID, model string, metrics bandit.PerformanceMetrics) error {
	// Delegate to state manager for persistent bandit arm updates
	if cr.stateManager != nil {
		if err := cr.stateManager.UpdateArm(model, metrics); err != nil {
			logrus.WithError(err).WithField("model", model).Error("Failed to update state manager arm")
			// Continue with local update as fallback
		}
	}

	// Also maintain local arms for backward compatibility and fallback
	cr.mu.Lock()
	defer cr.mu.Unlock()

	arm, exists := cr.globalArms[model]
	if !exists {
		arm = &bandit.BanditArm{
			Model:       model,
			LastUpdated: time.Now().Format(time.RFC3339),
		}
		cr.globalArms[model] = arm
	}

	arm.RequestCount++
	if metrics.Success {
		arm.SuccessCount++
	}
	arm.TotalLatency += metrics.Latency
	arm.TotalCost += metrics.Cost
	if metrics.FeedbackScore > 0 {
		arm.TotalFeedback += metrics.FeedbackScore
		arm.FeedbackCount++
	}
	arm.LastUpdated = time.Now().Format(time.RFC3339)

	logrus.WithFields(logrus.Fields{
		"model":      model,
		"request_id": requestID,
		"success":    metrics.Success,
		"latency":    metrics.Latency,
		"cost":       metrics.Cost,
		"feedback":   metrics.FeedbackScore,
	}).Debug("Updated bandit arm performance")

	return nil
}

// loadGlobalArms loads global bandit state from database
func (cr *ContextualRouter) loadGlobalArms() error {
	// This would query aggregated statistics from the database
	// For now, we'll start with empty state and build it from incoming updates
	logrus.Info("Global bandit arms initialized")
	return nil
}

// Health checks if the router is healthy
func (cr *ContextualRouter) Health(ctx context.Context) error {
	return cr.checkHealth()
}

// checkHealth performs internal health checks
func (cr *ContextualRouter) checkHealth() error {
	cr.mu.RLock()
	defer cr.mu.RUnlock()

	if cr.closed {
		return fmt.Errorf("contextual router is closed")
	}

	if cr.embeddingService == nil {
		return fmt.Errorf("embedding service not available")
	}

	return cr.embeddingService.Health(context.Background())
}

// Close releases resources
func (cr *ContextualRouter) Close() error {
	cr.mu.Lock()
	defer cr.mu.Unlock()

	if cr.closed {
		return nil
	}

	cr.closed = true
	logrus.Info("Contextual router closed")
	return nil
}

// tryContextualSelection attempts contextual bandit selection using embedding similarity
func (cr *ContextualRouter) tryContextualSelection(ctx context.Context, req *chat.Request) (model string, confidence float64, err error) {
	// Generate embedding for the request
	embedding, err := cr.embeddingService.EmbedMessages(ctx, req.Messages)
	if err != nil {
		return "", 0, fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Find top-K similar requests
	similarRequests, err := cr.findSimilarRequests(ctx, embedding)
	if err != nil {
		return "", 0, fmt.Errorf("failed to find similar requests: %w", err)
	}

	// Check if we have enough similar requests for confident decision
	if len(similarRequests) < cr.config.MinSimilarRequests {
		return "", 0, fmt.Errorf("insufficient similar requests: %d < %d", len(similarRequests), cr.config.MinSimilarRequests)
	}

	// Perform Thompson sampling on similar requests
	selectedModel := cr.thompsonSampleSimilar(similarRequests)

	// Calculate confidence based on number of similar requests and their quality
	confidence = cr.calculateContextualConfidence(similarRequests)

	return selectedModel, confidence, nil
}

// tryGlobalBanditSelection attempts global bandit selection using state manager
func (cr *ContextualRouter) tryGlobalBanditSelection() (model string, confidence float64, err error) {
	if cr.stateManager == nil {
		return "", 0, fmt.Errorf("state manager not available")
	}

	arms := cr.stateManager.GetAllArms()
	if len(arms) == 0 {
		return "", 0, fmt.Errorf("no global bandit arms available")
	}

	// Filter arms with sufficient data
	eligibleArms := make(map[string]*bandit.BanditArm)
	for model, arm := range arms {
		if arm.RequestCount >= cr.config.MinRequestsForGlobal {
			eligibleArms[model] = arm
		}
	}

	if len(eligibleArms) == 0 {
		return "", 0, fmt.Errorf("no arms with sufficient data (%d requests required)", cr.config.MinRequestsForGlobal)
	}

	// Perform Thompson sampling on global arms
	selectedModel := cr.thompsonSampleGlobalArms(eligibleArms)

	// Calculate confidence based on arm statistics
	confidence = cr.calculateGlobalConfidence(eligibleArms[selectedModel])

	return selectedModel, confidence, nil
}

// selectOptimisticExploration selects a model optimistically for cold start scenarios
func (cr *ContextualRouter) selectOptimisticExploration() string {
	if cr.stateManager == nil {
		return ""
	}

	// Get all arms and find models with little or no data
	arms := cr.stateManager.GetAllArms()

	// Look for models with very few requests (exploration candidates)
	explorationCandidates := make([]string, 0)
	for model, arm := range arms {
		if arm.RequestCount < cr.config.MinRequestsForGlobal {
			explorationCandidates = append(explorationCandidates, model)
		}
	}

	// If we have exploration candidates, select one with optimistic bias
	if len(explorationCandidates) > 0 {
		return cr.selectWithOptimisticBias(explorationCandidates, arms)
	}

	return ""
}

// selectFallbackModel randomly selects from available models instead of using a fixed default
func (cr *ContextualRouter) selectFallbackModel() string {
	var availableModels []string

	// First, try to get models from the state manager (models that have been initialized)
	if cr.stateManager != nil {
		arms := cr.stateManager.GetAllArms()
		for model := range arms {
			availableModels = append(availableModels, model)
		}
	}

	// If no models available from state manager, use configured models as fallback
	if len(availableModels) == 0 {
		// This assumes the router has access to configured allowed models through the state manager
		// If not available, we'll use a hardcoded list of the known free models
		availableModels = []string{
			"meta-llama/llama-3.2-3b-instruct:free",
			"google/gemma-2-9b-it:free",
			"microsoft/phi-3-mini-128k-instruct:free",
			"anthropic/claude-3.5-sonnet",
		}
	}

	// Randomly select from available models
	if len(availableModels) > 0 {
		selectedModel := availableModels[rand.Intn(len(availableModels))]
		logrus.WithFields(logrus.Fields{
			"selected_model":   selectedModel,
			"available_models": availableModels,
			"total_models":     len(availableModels),
		}).Info("Randomly selected fallback model")
		return selectedModel
	}

	// Ultimate fallback - this should rarely be reached
	return "anthropic/claude-3.5-sonnet"
}

// calculateContextualConfidence calculates confidence for contextual selection
func (cr *ContextualRouter) calculateContextualConfidence(similarRequests []bandit.SimilarRequest) float64 {
	if len(similarRequests) == 0 {
		return 0
	}

	// Base confidence on number of similar requests and their similarity scores
	countFactor := math.Min(1.0, float64(len(similarRequests))/float64(cr.config.MaxSimilarRequests))

	// Calculate average similarity
	var totalSimilarity float64
	for _, req := range similarRequests {
		totalSimilarity += req.Similarity
	}
	avgSimilarity := totalSimilarity / float64(len(similarRequests))

	// Combine factors
	confidence := countFactor * avgSimilarity
	return math.Min(1.0, confidence)
}

// calculateGlobalConfidence calculates confidence for global bandit selection
func (cr *ContextualRouter) calculateGlobalConfidence(arm *bandit.BanditArm) float64 {
	if arm == nil || arm.RequestCount == 0 {
		return 0
	}

	// Confidence based on number of requests (more data = higher confidence)
	requestFactor := math.Min(1.0, float64(arm.RequestCount)/100.0) // Max confidence at 100 requests

	// Adjust based on success rate consistency
	successRate := float64(arm.SuccessCount) / float64(arm.RequestCount)
	consistencyFactor := 1.0 - math.Abs(successRate-0.5)*2 // Higher confidence for more extreme success rates

	confidence := requestFactor * (0.7 + 0.3*consistencyFactor)
	return math.Min(1.0, confidence)
}

// selectWithOptimisticBias selects from candidates with optimistic exploration and random tie-breaking
func (cr *ContextualRouter) selectWithOptimisticBias(candidates []string, arms map[string]*bandit.BanditArm) string {
	if len(candidates) == 0 {
		return ""
	}

	type modelScore struct {
		model string
		score float64
	}

	var modelScores []modelScore
	bestScore := -1.0

	// Calculate scores for all candidates
	for _, model := range candidates {
		arm := arms[model]

		// Calculate optimistic score
		var score float64
		if arm == nil || arm.RequestCount == 0 {
			// Pure optimistic prior for completely new models
			score = cr.config.OptimisticPrior + cr.config.ExplorationBonus
		} else {
			// Optimistic adjustment for models with little data
			successRate := float64(arm.SuccessCount) / float64(arm.RequestCount)
			explorationBonus := cr.config.ExplorationBonus / math.Max(1, float64(arm.RequestCount)/10.0)
			score = successRate + explorationBonus
		}

		modelScores = append(modelScores, modelScore{model: model, score: score})
		if score > bestScore {
			bestScore = score
		}
	}

	// Collect all models with the best score for random tie-breaking
	var bestModels []string
	const epsilon = 1e-9 // Small epsilon for float comparison
	for _, ms := range modelScores {
		if math.Abs(ms.score-bestScore) < epsilon {
			bestModels = append(bestModels, ms.model)
		}
	}

	// Randomly select from the best models
	if len(bestModels) > 0 {
		selectedModel := bestModels[rand.Intn(len(bestModels))]
		if len(bestModels) > 1 {
			logrus.WithFields(logrus.Fields{
				"selected_model":   selectedModel,
				"tied_models":      bestModels,
				"best_score":       bestScore,
				"total_candidates": len(candidates),
			}).Debug("Random tie-breaking in optimistic exploration")
		}
		return selectedModel
	}

	// Fallback (should not happen if candidates is not empty)
	return candidates[0]
}

// thompsonSampleGlobalArms performs Thompson sampling on global bandit arms
func (cr *ContextualRouter) thompsonSampleGlobalArms(arms map[string]*bandit.BanditArm) string {
	if len(arms) == 0 {
		return ""
	}

	type modelScore struct {
		model string
		score float64
	}

	var scores []modelScore
	for model, arm := range arms {
		if arm.RequestCount == 0 {
			// Use optimistic prior for new arms
			sample := cr.config.OptimisticPrior + cr.config.ExplorationBonus
			scores = append(scores, modelScore{model: model, score: sample})
			continue
		}

		// Use Beta distribution based on success/failure counts
		alpha := math.Max(1, float64(arm.SuccessCount)+1)
		beta := math.Max(1, float64(arm.RequestCount-arm.SuccessCount)+1)

		sample := cr.betaSample(alpha, beta)
		scores = append(scores, modelScore{model: model, score: sample})
	}

	if len(scores) == 0 {
		return ""
	}

	// Sort by score and select the highest
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	return scores[0].model
}
