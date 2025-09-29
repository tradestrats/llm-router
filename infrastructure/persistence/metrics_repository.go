package persistence

import (
	"context"
	"fmt"

	"llm-router/domain/persistence"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

// MetricsRepository implements persistence.MetricsRepository
type MetricsRepository struct {
	db *gorm.DB
}

// NewMetricsRepository creates a new metrics repository
func NewMetricsRepository(db *gorm.DB) persistence.MetricsRepository {
	return &MetricsRepository{db: db}
}

// getDB returns the database instance, checking for transaction context
func (r *MetricsRepository) getDB(ctx context.Context) *gorm.DB {
	if tx, ok := ctx.Value("gorm_tx").(*gorm.DB); ok && tx != nil {
		return tx
	}
	return r.db.WithContext(ctx)
}

// Create creates a new metrics record
func (r *MetricsRepository) Create(ctx context.Context, entity *persistence.RequestMetrics) error {
	db := r.getDB(ctx)
	if err := db.Create(entity).Error; err != nil {
		return fmt.Errorf("failed to create metrics record: %w", err)
	}
	return nil
}

// Update updates an existing metrics record
func (r *MetricsRepository) Update(ctx context.Context, entity *persistence.RequestMetrics) error {
	db := r.getDB(ctx)
	if err := db.Save(entity).Error; err != nil {
		return fmt.Errorf("failed to update metrics record: %w", err)
	}
	return nil
}

// FindByID finds a metrics record by ID
func (r *MetricsRepository) FindByID(ctx context.Context, id uuid.UUID) (*persistence.RequestMetrics, error) {
	db := r.getDB(ctx)
	var record persistence.RequestMetrics
	if err := db.First(&record, "id = ?", id).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, fmt.Errorf("metrics record not found: %w", err)
		}
		return nil, fmt.Errorf("failed to find metrics record: %w", err)
	}
	return &record, nil
}

// FindByRequestID finds metrics by request ID
func (r *MetricsRepository) FindByRequestID(ctx context.Context, requestID uuid.UUID) (*persistence.RequestMetrics, error) {
	db := r.getDB(ctx)
	var record persistence.RequestMetrics
	if err := db.First(&record, "request_id = ?", requestID).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, fmt.Errorf("metrics record not found for request: %w", err)
		}
		return nil, fmt.Errorf("failed to find metrics record by request ID: %w", err)
	}
	return &record, nil
}

// CreateOrUpdate creates a new metrics record or updates existing one
func (r *MetricsRepository) CreateOrUpdate(ctx context.Context, metrics *persistence.RequestMetrics) error {
	db := r.getDB(ctx)

	// Try to find existing record
	var existing persistence.RequestMetrics
	err := db.First(&existing, "request_id = ?", metrics.RequestID).Error

	if err == gorm.ErrRecordNotFound {
		// Create new record
		if err := db.Create(metrics).Error; err != nil {
			return fmt.Errorf("failed to create metrics record: %w", err)
		}
		return nil
	} else if err != nil {
		return fmt.Errorf("failed to check existing metrics: %w", err)
	}

	// Update existing record
	existing.TotalCost = metrics.TotalCost
	existing.TokensUsed = metrics.TokensUsed
	existing.LatencyMs = metrics.LatencyMs

	if err := db.Save(&existing).Error; err != nil {
		return fmt.Errorf("failed to update existing metrics: %w", err)
	}

	// Copy back the ID for the caller
	metrics.ID = existing.ID
	return nil
}

// GetAggregatedMetrics returns aggregated metrics across all requests
func (r *MetricsRepository) GetAggregatedMetrics(ctx context.Context, limit int) (*persistence.AggregatedMetrics, error) {
	db := r.getDB(ctx)

	var result struct {
		TotalRequests    int64   `json:"total_requests"`
		AverageCost      float64 `json:"average_cost"`
		AverageTokens    float64 `json:"average_tokens"`
		AverageLatencyMs float64 `json:"average_latency_ms"`
		TotalCost        float64 `json:"total_cost"`
		TotalTokens      int64   `json:"total_tokens"`
	}

	query := db.Model(&persistence.RequestMetrics{}).
		Select(`
			COUNT(*) as total_requests,
			COALESCE(AVG(total_cost), 0) as average_cost,
			COALESCE(AVG(tokens_used), 0) as average_tokens,
			COALESCE(AVG(latency_ms), 0) as average_latency_ms,
			COALESCE(SUM(total_cost), 0) as total_cost,
			COALESCE(SUM(tokens_used), 0) as total_tokens
		`)

	if limit > 0 {
		// Get metrics for the most recent requests
		subQuery := db.Model(&persistence.RequestMetrics{}).
			Select("request_id").
			Order("created_at DESC").
			Limit(limit)
		query = query.Where("request_id IN (?)", subQuery)
	}

	if err := query.Scan(&result).Error; err != nil {
		return nil, fmt.Errorf("failed to get aggregated metrics: %w", err)
	}

	// Get average feedback score
	var avgFeedback float64
	feedbackQuery := db.Model(&persistence.RequestFeedback{}).
		Select("COALESCE(AVG(score), 0)")

	if limit > 0 {
		// Match the same request limitation
		subQuery := db.Model(&persistence.RequestMetrics{}).
			Select("request_id").
			Order("created_at DESC").
			Limit(limit)
		feedbackQuery = feedbackQuery.Where("request_id IN (?)", subQuery)
	}

	if err := feedbackQuery.Scan(&avgFeedback).Error; err != nil {
		return nil, fmt.Errorf("failed to get average feedback: %w", err)
	}

	return &persistence.AggregatedMetrics{
		TotalRequests:    result.TotalRequests,
		AverageCost:      result.AverageCost,
		AverageTokens:    result.AverageTokens,
		AverageLatencyMs: result.AverageLatencyMs,
		AverageFeedback:  avgFeedback,
		TotalCost:        result.TotalCost,
		TotalTokens:      result.TotalTokens,
	}, nil
}

// Delete deletes a metrics record
func (r *MetricsRepository) Delete(ctx context.Context, id uuid.UUID) error {
	db := r.getDB(ctx)
	result := db.Delete(&persistence.RequestMetrics{}, "id = ?", id)
	if result.Error != nil {
		return fmt.Errorf("failed to delete metrics record: %w", result.Error)
	}
	if result.RowsAffected == 0 {
		return fmt.Errorf("metrics record not found for deletion")
	}
	return nil
}
