package bandit

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"llm-router/domain/bandit"
	"llm-router/domain/persistence"

	"github.com/sirupsen/logrus"
	"gorm.io/gorm"
)

// StateManager manages bandit state by computing statistics on-demand from request data
//
// SIMPLIFIED APPROACH:
// - Added in-memory caching to reduce database load
type StateManager struct {
	db          *gorm.DB
	requestRepo persistence.RequestRepository
	mu          sync.RWMutex
	closed      int64

	// Caching fields
	allArmsCache       map[string]*bandit.BanditArm
	allArmsCacheExpiry time.Time
	cacheTTL           time.Duration
	cacheMu            sync.RWMutex

	// Cache statistics
	cacheHits   int64
	cacheMisses int64

	// Background refresh
	refreshContext context.Context
	refreshCancel  context.CancelFunc
	refreshWG      sync.WaitGroup
}

// NewStateManager creates a new simplified bandit state manager
func NewStateManager(db *gorm.DB, requestRepo persistence.RequestRepository) (*StateManager, error) {
	ctx, cancel := context.WithCancel(context.Background())

	sm := &StateManager{
		db:          db,
		requestRepo: requestRepo,

		// Initialize cache with 30-second TTL
		allArmsCache:   make(map[string]*bandit.BanditArm),
		cacheTTL:       30 * time.Second,
		refreshContext: ctx,
		refreshCancel:  cancel,
	}

	// Start background cache refresh
	sm.startBackgroundRefresh()

	logrus.Info("Bandit state manager initialized with 30s TTL cache")
	return sm, nil
}

// GetArm computes bandit arm statistics on-demand from request data
func (sm *StateManager) GetArm(model string) *bandit.BanditArm {
	if atomic.LoadInt64(&sm.closed) == 1 {
		return nil
	}

	sm.mu.RLock()
	defer sm.mu.RUnlock()

	// Compute statistics from request data
	arm := sm.computeArmStatistics(model)
	return arm
}

// GetAllArms computes statistics for all models using cached data when possible
func (sm *StateManager) GetAllArms() map[string]*bandit.BanditArm {
	if atomic.LoadInt64(&sm.closed) == 1 {
		return make(map[string]*bandit.BanditArm)
	}

	// Check cache first
	sm.cacheMu.RLock()
	if time.Now().Before(sm.allArmsCacheExpiry) && sm.allArmsCache != nil {
		atomic.AddInt64(&sm.cacheHits, 1)
		cachedArms := make(map[string]*bandit.BanditArm, len(sm.allArmsCache))
		for k, v := range sm.allArmsCache {
			// Return deep copy to prevent external mutations
			armCopy := *v
			cachedArms[k] = &armCopy
		}
		sm.cacheMu.RUnlock()

		logrus.WithFields(logrus.Fields{
			"models_count": len(cachedArms),
			"cache_hit":    true,
		}).Debug("Returned cached bandit arms")

		return cachedArms
	}
	sm.cacheMu.RUnlock()

	// Cache miss - compute fresh data
	atomic.AddInt64(&sm.cacheMisses, 1)
	arms := sm.computeAllArmsFromDB()

	// Update cache
	sm.cacheMu.Lock()
	sm.allArmsCache = make(map[string]*bandit.BanditArm, len(arms))
	for k, v := range arms {
		// Store deep copy in cache
		armCopy := *v
		sm.allArmsCache[k] = &armCopy
	}
	sm.allArmsCacheExpiry = time.Now().Add(sm.cacheTTL)
	sm.cacheMu.Unlock()

	logrus.WithFields(logrus.Fields{
		"models_count": len(arms),
		"cache_hit":    false,
		"cache_ttl":    sm.cacheTTL.Seconds(),
	}).Debug("Computed and cached fresh bandit arms")

	return arms
}

// computeAllArmsFromDB performs the actual database query for all bandit arms
func (sm *StateManager) computeAllArmsFromDB() map[string]*bandit.BanditArm {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	queryStart := time.Now()

	// Single optimized query for all models at once
	var results []struct {
		Model         string  `gorm:"column:model"`
		RequestCount  int     `gorm:"column:request_count"`
		SuccessCount  int     `gorm:"column:success_count"`
		TotalLatency  float64 `gorm:"column:total_latency"`
		TotalCost     float64 `gorm:"column:total_cost"`
		TotalFeedback float64 `gorm:"column:total_feedback"`
		FeedbackCount int     `gorm:"column:feedback_count"`
	}

	err := sm.db.WithContext(context.Background()).Raw(`
		SELECT
			r.model,
			COUNT(r.id) as request_count,
			COUNT(r.id) as success_count,  -- All completed requests are successful
			COALESCE(SUM(m.latency_ms), 0) as total_latency,
			COALESCE(SUM(m.total_cost), 0) as total_cost,
			COALESCE(SUM(f.score), 0) as total_feedback,
			COUNT(f.id) as feedback_count
		FROM requests r
		LEFT JOIN request_metrics m ON r.id = m.request_id
		LEFT JOIN request_feedback f ON r.id = f.request_id AND f.score > 0
		WHERE r.status = ?
		GROUP BY r.model
		ORDER BY request_count DESC
	`, persistence.RequestStatusCompleted).Scan(&results).Error

	queryDuration := time.Since(queryStart)

	if err != nil {
		logrus.WithError(err).WithFields(logrus.Fields{
			"query_duration": queryDuration.Milliseconds(),
		}).Error("Failed to get all bandit arms from database")
		return make(map[string]*bandit.BanditArm)
	}

	arms := make(map[string]*bandit.BanditArm, len(results))
	totalRequests := 0
	for _, result := range results {
		arms[result.Model] = &bandit.BanditArm{
			Model:         result.Model,
			RequestCount:  result.RequestCount,
			SuccessCount:  result.SuccessCount,
			TotalLatency:  result.TotalLatency,
			TotalCost:     result.TotalCost,
			TotalFeedback: result.TotalFeedback,
			FeedbackCount: result.FeedbackCount,
			LastUpdated:   time.Now().Format(time.RFC3339),
		}
		totalRequests += result.RequestCount
	}

	logrus.WithFields(logrus.Fields{
		"models_count":   len(arms),
		"total_requests": totalRequests,
		"query_duration": queryDuration.Milliseconds(),
	}).Debug("Computed all bandit arms from database")

	// Log slow queries for performance monitoring
	if queryDuration > 200*time.Millisecond {
		logrus.WithFields(logrus.Fields{
			"models_count":   len(arms),
			"total_requests": totalRequests,
			"query_duration": queryDuration.Milliseconds(),
		}).Warn("Slow GetAllArms query detected - consider index optimization")
	}

	return arms
}

// UpdateArm is a no-op in the simplified approach - statistics are computed on-demand
// This method exists for interface compatibility but doesn't need to do anything
func (sm *StateManager) UpdateArm(model string, metrics bandit.PerformanceMetrics) error {
	if atomic.LoadInt64(&sm.closed) == 1 {
		return fmt.Errorf("state manager is closed")
	}

	// No-op: statistics are computed on-demand from request data
	// The request data is already stored in the requests table by the RequestTracker
	logrus.WithFields(logrus.Fields{
		"model":    model,
		"success":  metrics.Success,
		"latency":  metrics.Latency,
		"cost":     metrics.Cost,
		"feedback": metrics.FeedbackScore,
	}).Debug("Bandit arm update (no-op: statistics computed on-demand)")

	return nil
}

// CreateArm is a no-op in the simplified approach
func (sm *StateManager) CreateArm(model string) error {
	if atomic.LoadInt64(&sm.closed) == 1 {
		return fmt.Errorf("state manager is closed")
	}

	// No-op: arms are created automatically when requests are processed
	logrus.WithField("model", model).Debug("Bandit arm creation (no-op: arms created on-demand)")
	return nil
}

// GetStatistics returns aggregated statistics computed on-demand
func (sm *StateManager) GetStatistics() map[string]interface{} {
	if atomic.LoadInt64(&sm.closed) == 1 {
		return map[string]interface{}{"error": "state manager is closed"}
	}

	sm.mu.RLock()
	defer sm.mu.RUnlock()

	allArms := sm.GetAllArms()

	stats := map[string]interface{}{
		"total_arms":      len(allArms),
		"total_requests":  0,
		"total_successes": 0,
		"models":          make(map[string]interface{}),
		"cache":           sm.GetCacheStatistics(),
	}

	var totalRequests, totalSuccesses int
	modelStats := make(map[string]interface{})

	for model, arm := range allArms {
		totalRequests += arm.RequestCount
		totalSuccesses += arm.SuccessCount

		var avgFeedback float64
		if arm.FeedbackCount > 0 {
			avgFeedback = arm.TotalFeedback / float64(arm.FeedbackCount)
		}

		modelStats[model] = map[string]interface{}{
			"request_count": arm.RequestCount,
			"success_count": arm.SuccessCount,
			"success_rate":  float64(arm.SuccessCount) / float64(arm.RequestCount),
			"avg_latency":   arm.TotalLatency / float64(arm.RequestCount),
			"avg_cost":      arm.TotalCost / float64(arm.RequestCount),
			"avg_feedback":  avgFeedback,
			"last_updated":  arm.LastUpdated,
		}
	}

	stats["total_requests"] = totalRequests
	stats["total_successes"] = totalSuccesses
	stats["models"] = modelStats

	return stats
}

// Health checks if the state manager is healthy
func (sm *StateManager) Health() error {
	if atomic.LoadInt64(&sm.closed) == 1 {
		return fmt.Errorf("state manager is closed")
	}

	// Simple health check: try to query the database
	var count int64
	err := sm.db.WithContext(context.Background()).
		Model(&persistence.RequestRecord{}).
		Count(&count).Error

	if err != nil {
		return fmt.Errorf("database health check failed: %w", err)
	}

	return nil
}

// Close shuts down the state manager
func (sm *StateManager) Close() error {
	if !atomic.CompareAndSwapInt64(&sm.closed, 0, 1) {
		return fmt.Errorf("state manager already closed")
	}

	// Stop background refresh
	if sm.refreshCancel != nil {
		sm.refreshCancel()
		sm.refreshWG.Wait()
	}

	logrus.Info("Bandit state manager closed")
	return nil
}

// startBackgroundRefresh starts a goroutine to proactively refresh the cache
func (sm *StateManager) startBackgroundRefresh() {
	sm.refreshWG.Add(1)
	go func() {
		defer sm.refreshWG.Done()

		// Refresh every 25 seconds (5 seconds before cache expires)
		ticker := time.NewTicker(25 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				if atomic.LoadInt64(&sm.closed) == 1 {
					return
				}

				// Check if cache will expire soon
				sm.cacheMu.RLock()
				willExpireSoon := time.Now().Add(10 * time.Second).After(sm.allArmsCacheExpiry)
				sm.cacheMu.RUnlock()

				if willExpireSoon {
					sm.refreshCacheAsync()
				}

			case <-sm.refreshContext.Done():
				logrus.Debug("Background cache refresh stopped")
				return
			}
		}
	}()

	logrus.Debug("Background cache refresh started")
}

// refreshCacheAsync refreshes the cache in the background
func (sm *StateManager) refreshCacheAsync() {
	go func() {
		if atomic.LoadInt64(&sm.closed) == 1 {
			return
		}

		logrus.Debug("Starting background cache refresh")
		startTime := time.Now()

		arms := sm.computeAllArmsFromDB()
		if len(arms) == 0 {
			logrus.Warn("Background cache refresh returned no arms")
			return
		}

		// Update cache
		sm.cacheMu.Lock()
		sm.allArmsCache = make(map[string]*bandit.BanditArm, len(arms))
		for k, v := range arms {
			armCopy := *v
			sm.allArmsCache[k] = &armCopy
		}
		sm.allArmsCacheExpiry = time.Now().Add(sm.cacheTTL)
		sm.cacheMu.Unlock()

		logrus.WithFields(logrus.Fields{
			"models_count":     len(arms),
			"refresh_duration": time.Since(startTime).Milliseconds(),
		}).Debug("Background cache refresh completed")
	}()
}

// GetCacheStatistics returns cache performance metrics
func (sm *StateManager) GetCacheStatistics() map[string]interface{} {
	hits := atomic.LoadInt64(&sm.cacheHits)
	misses := atomic.LoadInt64(&sm.cacheMisses)
	total := hits + misses

	var hitRatio float64
	if total > 0 {
		hitRatio = float64(hits) / float64(total)
	}

	sm.cacheMu.RLock()
	cacheSize := len(sm.allArmsCache)
	timeToExpiry := sm.allArmsCacheExpiry.Sub(time.Now())
	sm.cacheMu.RUnlock()

	return map[string]interface{}{
		"cache_hits":        hits,
		"cache_misses":      misses,
		"cache_hit_ratio":   hitRatio,
		"cache_size":        cacheSize,
		"cache_ttl_seconds": sm.cacheTTL.Seconds(),
		"time_to_expiry":    timeToExpiry.Seconds(),
	}
}

// computeArmStatistics computes bandit arm statistics using efficient database aggregation
func (sm *StateManager) computeArmStatistics(model string) *bandit.BanditArm {
	ctx := context.Background()
	queryStart := time.Now()

	// Single optimized query with JOINs and aggregation
	var result struct {
		RequestCount  int     `gorm:"column:request_count"`
		SuccessCount  int     `gorm:"column:success_count"`
		TotalLatency  float64 `gorm:"column:total_latency"`
		TotalCost     float64 `gorm:"column:total_cost"`
		TotalFeedback float64 `gorm:"column:total_feedback"`
		FeedbackCount int     `gorm:"column:feedback_count"`
	}

	err := sm.db.WithContext(ctx).Raw(`
		SELECT
			COUNT(r.id) as request_count,
			COUNT(r.id) as success_count,  -- All completed requests are successful
			COALESCE(SUM(m.latency_ms), 0) as total_latency,
			COALESCE(SUM(m.total_cost), 0) as total_cost,
			COALESCE(SUM(f.score), 0) as total_feedback,
			COUNT(f.id) as feedback_count
		FROM requests r
		LEFT JOIN request_metrics m ON r.id = m.request_id
		LEFT JOIN request_feedback f ON r.id = f.request_id AND f.score > 0
		WHERE r.model = ? AND r.status = ?
	`, model, persistence.RequestStatusCompleted).Scan(&result).Error

	queryDuration := time.Since(queryStart)

	if err != nil {
		logrus.WithError(err).WithFields(logrus.Fields{
			"model":          model,
			"query_duration": queryDuration.Milliseconds(),
		}).Error("Failed to compute arm statistics")
		return nil
	}

	if result.RequestCount == 0 {
		logrus.WithFields(logrus.Fields{
			"model":          model,
			"query_duration": queryDuration.Milliseconds(),
		}).Debug("No completed requests found for model")
		return nil // No completed requests for this model
	}

	arm := &bandit.BanditArm{
		Model:         model,
		RequestCount:  result.RequestCount,
		SuccessCount:  result.SuccessCount,
		TotalLatency:  result.TotalLatency,
		TotalCost:     result.TotalCost,
		TotalFeedback: result.TotalFeedback,
		FeedbackCount: result.FeedbackCount,
		LastUpdated:   time.Now().Format(time.RFC3339),
	}

	logrus.WithFields(logrus.Fields{
		"model":          model,
		"request_count":  arm.RequestCount,
		"success_count":  arm.SuccessCount,
		"avg_latency":    arm.TotalLatency / float64(arm.RequestCount),
		"avg_cost":       arm.TotalCost / float64(arm.RequestCount),
		"query_duration": queryDuration.Milliseconds(),
	}).Debug("Computed bandit arm statistics with optimized query")

	// Log slow queries for performance monitoring
	if queryDuration > 100*time.Millisecond {
		logrus.WithFields(logrus.Fields{
			"model":          model,
			"query_duration": queryDuration.Milliseconds(),
			"request_count":  arm.RequestCount,
		}).Warn("Slow bandit arm query detected - consider index optimization")
	}

	return arm
}
