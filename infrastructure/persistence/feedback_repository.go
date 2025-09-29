package persistence

import (
	"context"
	"fmt"

	"llm-router/domain/persistence"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

// FeedbackRepository implements persistence.FeedbackRepository
type FeedbackRepository struct {
	db *gorm.DB
}

// NewFeedbackRepository creates a new feedback repository
func NewFeedbackRepository(db *gorm.DB) persistence.FeedbackRepository {
	return &FeedbackRepository{db: db}
}

// getDB returns the database instance, checking for transaction context
func (r *FeedbackRepository) getDB(ctx context.Context) *gorm.DB {
	if tx, ok := ctx.Value("gorm_tx").(*gorm.DB); ok && tx != nil {
		return tx
	}
	return r.db.WithContext(ctx)
}

// Create creates a new feedback record
func (r *FeedbackRepository) Create(ctx context.Context, entity *persistence.RequestFeedback) error {
	db := r.getDB(ctx)
	if err := db.Create(entity).Error; err != nil {
		return fmt.Errorf("failed to create feedback record: %w", err)
	}
	return nil
}

// Update updates an existing feedback record
func (r *FeedbackRepository) Update(ctx context.Context, entity *persistence.RequestFeedback) error {
	db := r.getDB(ctx)
	if err := db.Save(entity).Error; err != nil {
		return fmt.Errorf("failed to update feedback record: %w", err)
	}
	return nil
}

// FindByID finds a feedback record by ID
func (r *FeedbackRepository) FindByID(ctx context.Context, id uuid.UUID) (*persistence.RequestFeedback, error) {
	db := r.getDB(ctx)
	var record persistence.RequestFeedback
	if err := db.First(&record, "id = ?", id).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, fmt.Errorf("feedback record not found: %w", err)
		}
		return nil, fmt.Errorf("failed to find feedback record: %w", err)
	}
	return &record, nil
}

// FindByRequestID finds all feedback records for a specific request
func (r *FeedbackRepository) FindByRequestID(ctx context.Context, requestID uuid.UUID) ([]*persistence.RequestFeedback, error) {
	db := r.getDB(ctx)
	var records []*persistence.RequestFeedback
	if err := db.Where("request_id = ?", requestID).Order("created_at DESC").Find(&records).Error; err != nil {
		return nil, fmt.Errorf("failed to find feedback records by request ID: %w", err)
	}
	return records, nil
}

// GetAverageScore calculates the average feedback score
// If requestID is provided, calculates average for that specific request
// If requestID is nil, calculates overall average across all requests
func (r *FeedbackRepository) GetAverageScore(ctx context.Context, requestID *uuid.UUID) (float64, error) {
	db := r.getDB(ctx)

	query := db.Model(&persistence.RequestFeedback{})
	if requestID != nil {
		query = query.Where("request_id = ?", *requestID)
	}

	var avgScore float64
	if err := query.Select("COALESCE(AVG(score), 0)").Scan(&avgScore).Error; err != nil {
		return 0, fmt.Errorf("failed to calculate average feedback score: %w", err)
	}

	return avgScore, nil
}

// FindRecentFeedback finds recent feedback records across all requests
func (r *FeedbackRepository) FindRecentFeedback(ctx context.Context, limit int) ([]*persistence.RequestFeedback, error) {
	db := r.getDB(ctx)
	var records []*persistence.RequestFeedback

	query := db.Order("created_at DESC")
	if limit > 0 {
		query = query.Limit(limit)
	}

	if err := query.Find(&records).Error; err != nil {
		return nil, fmt.Errorf("failed to find recent feedback records: %w", err)
	}

	return records, nil
}

// Delete deletes a feedback record
func (r *FeedbackRepository) Delete(ctx context.Context, id uuid.UUID) error {
	db := r.getDB(ctx)
	result := db.Delete(&persistence.RequestFeedback{}, "id = ?", id)
	if result.Error != nil {
		return fmt.Errorf("failed to delete feedback record: %w", result.Error)
	}
	if result.RowsAffected == 0 {
		return fmt.Errorf("feedback record not found for deletion")
	}
	return nil
}
