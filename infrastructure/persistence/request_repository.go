package persistence

import (
	"context"
	"fmt"

	"llm-router/domain/persistence"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

// RequestRepository implements persistence.RequestRepository
type RequestRepository struct {
	db *gorm.DB
}

// NewRequestRepository creates a new request repository
func NewRequestRepository(db *gorm.DB) persistence.RequestRepository {
	return &RequestRepository{db: db}
}

// getDB returns the database instance, checking for transaction context
func (r *RequestRepository) getDB(ctx context.Context) *gorm.DB {
	if tx, ok := ctx.Value("gorm_tx").(*gorm.DB); ok && tx != nil {
		return tx
	}
	return r.db.WithContext(ctx)
}

// Create creates a new request record
func (r *RequestRepository) Create(ctx context.Context, entity *persistence.RequestRecord) error {
	db := r.getDB(ctx)
	if err := db.Create(entity).Error; err != nil {
		return fmt.Errorf("failed to create request record: %w", err)
	}
	return nil
}

// Update updates an existing request record
func (r *RequestRepository) Update(ctx context.Context, entity *persistence.RequestRecord) error {
	db := r.getDB(ctx)
	if err := db.Save(entity).Error; err != nil {
		return fmt.Errorf("failed to update request record: %w", err)
	}
	return nil
}

// FindByID finds a request record by ID
func (r *RequestRepository) FindByID(ctx context.Context, id uuid.UUID) (*persistence.RequestRecord, error) {
	db := r.getDB(ctx)
	var record persistence.RequestRecord
	if err := db.First(&record, "id = ?", id).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, fmt.Errorf("request record not found: %w", err)
		}
		return nil, fmt.Errorf("failed to find request record: %w", err)
	}
	return &record, nil
}

// FindByIDWithRelations finds a request record with its related metrics and feedback
func (r *RequestRepository) FindByIDWithRelations(ctx context.Context, id uuid.UUID) (*persistence.RequestRecord, error) {
	db := r.getDB(ctx)
	var record persistence.RequestRecord
	if err := db.Preload("Metrics").Preload("Feedback").First(&record, "id = ?", id).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, fmt.Errorf("request record not found: %w", err)
		}
		return nil, fmt.Errorf("failed to find request record with relations: %w", err)
	}
	return &record, nil
}

// FindByStatus finds request records by status
func (r *RequestRepository) FindByStatus(ctx context.Context, status persistence.RequestStatus, limit int) ([]*persistence.RequestRecord, error) {
	db := r.getDB(ctx)
	var records []*persistence.RequestRecord
	query := db.Where("status = ?", status).Order("created_at DESC")
	if limit > 0 {
		query = query.Limit(limit)
	}
	if err := query.Find(&records).Error; err != nil {
		return nil, fmt.Errorf("failed to find request records by status: %w", err)
	}
	return records, nil
}

// FindRecent finds recent request records
func (r *RequestRepository) FindRecent(ctx context.Context, limit int) ([]*persistence.RequestRecord, error) {
	db := r.getDB(ctx)
	var records []*persistence.RequestRecord
	query := db.Order("created_at DESC")
	if limit > 0 {
		query = query.Limit(limit)
	}
	if err := query.Find(&records).Error; err != nil {
		return nil, fmt.Errorf("failed to find recent request records: %w", err)
	}
	return records, nil
}

// UpdateStatus updates the status of a request record
func (r *RequestRepository) UpdateStatus(ctx context.Context, id uuid.UUID, status persistence.RequestStatus) error {
	db := r.getDB(ctx)
	result := db.Model(&persistence.RequestRecord{}).Where("id = ?", id).Update("status", status)
	if result.Error != nil {
		return fmt.Errorf("failed to update request status: %w", result.Error)
	}
	if result.RowsAffected == 0 {
		return fmt.Errorf("request record not found for status update")
	}
	return nil
}

// Delete deletes a request record
func (r *RequestRepository) Delete(ctx context.Context, id uuid.UUID) error {
	db := r.getDB(ctx)
	result := db.Delete(&persistence.RequestRecord{}, "id = ?", id)
	if result.Error != nil {
		return fmt.Errorf("failed to delete request record: %w", result.Error)
	}
	if result.RowsAffected == 0 {
		return fmt.Errorf("request record not found for deletion")
	}
	return nil
}
