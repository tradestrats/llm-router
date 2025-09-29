package persistence

import (
	"encoding/json"
	"time"

	"github.com/google/uuid"
	"github.com/pgvector/pgvector-go"
	"gorm.io/gorm"
)

// RequestRecord stores the complete request and response data
type RequestRecord struct {
	ID           uuid.UUID        `gorm:"type:uuid;primary_key;default:gen_random_uuid()" json:"id"`
	RequestData  json.RawMessage  `gorm:"type:jsonb;not null" json:"request_data"`
	ResponseData json.RawMessage  `gorm:"type:jsonb" json:"response_data,omitempty"`
	Model        string           `gorm:"type:varchar(255);not null;index" json:"model"`
	IsStreaming  bool             `gorm:"default:false;index" json:"is_streaming"`
	Status       RequestStatus    `gorm:"type:varchar(50);not null;default:'pending';index" json:"status"`
	Embedding    *pgvector.Vector `gorm:"type:vector(384)" json:"embedding,omitempty"`
	CreatedAt    time.Time        `gorm:"autoCreateTime" json:"created_at"`
	UpdatedAt    time.Time        `gorm:"autoUpdateTime" json:"updated_at"`

	// Relations
	Metrics  *RequestMetrics   `gorm:"foreignKey:RequestID;constraint:OnDelete:CASCADE" json:"metrics,omitempty"`
	Feedback []RequestFeedback `gorm:"foreignKey:RequestID;constraint:OnDelete:CASCADE" json:"feedback,omitempty"`
}

// RequestStatus represents the status of a request
type RequestStatus string

const (
	RequestStatusPending   RequestStatus = "pending"
	RequestStatusCompleted RequestStatus = "completed"
	RequestStatusFailed    RequestStatus = "failed"
)

// RequestMetrics stores performance and cost metrics for each request
type RequestMetrics struct {
	ID         uuid.UUID `gorm:"type:uuid;primary_key;default:gen_random_uuid()" json:"id"`
	RequestID  uuid.UUID `gorm:"type:uuid;not null;index" json:"request_id"`
	TotalCost  float64   `gorm:"type:decimal(10,6);default:0" json:"total_cost"`
	TokensUsed int       `gorm:"default:0" json:"tokens_used"`
	LatencyMs  int64     `gorm:"default:0" json:"latency_ms"`
	CreatedAt  time.Time `gorm:"autoCreateTime" json:"created_at"`
}

// RequestFeedback stores user feedback for each request
type RequestFeedback struct {
	ID           uuid.UUID `gorm:"type:uuid;primary_key;default:gen_random_uuid()" json:"id"`
	RequestID    uuid.UUID `gorm:"type:uuid;not null;index" json:"request_id"`
	FeedbackText string    `gorm:"type:text" json:"feedback_text"`
	Score        float64   `gorm:"type:decimal(3,2);check:score >= 0 AND score <= 1" json:"score"`
	CreatedAt    time.Time `gorm:"autoCreateTime" json:"created_at"`
}

// BeforeCreate hook for RequestRecord
func (r *RequestRecord) BeforeCreate(tx *gorm.DB) error {
	if r.ID == uuid.Nil {
		r.ID = uuid.New()
	}

	// Ensure embedding is never an empty slice - convert to nil if empty
	if r.Embedding != nil {
		// Check if the vector is empty by examining its underlying slice
		if len(r.Embedding.Slice()) == 0 {
			r.Embedding = nil
		}
	}

	return nil
}

// BeforeUpdate hook for RequestRecord
func (r *RequestRecord) BeforeUpdate(tx *gorm.DB) error {
	// Ensure embedding is never an empty slice - convert to nil if empty
	if r.Embedding != nil {
		// Check if the vector is empty by examining its underlying slice
		if len(r.Embedding.Slice()) == 0 {
			r.Embedding = nil
		}
	}

	return nil
}

// BeforeCreate hook for RequestMetrics
func (m *RequestMetrics) BeforeCreate(tx *gorm.DB) error {
	if m.ID == uuid.Nil {
		m.ID = uuid.New()
	}
	return nil
}

// BeforeCreate hook for RequestFeedback
func (f *RequestFeedback) BeforeCreate(tx *gorm.DB) error {
	if f.ID == uuid.Nil {
		f.ID = uuid.New()
	}
	return nil
}

// TableName returns the table name for RequestRecord
func (RequestRecord) TableName() string {
	return "requests"
}

// TableName returns the table name for RequestMetrics
func (RequestMetrics) TableName() string {
	return "request_metrics"
}

// TableName returns the table name for RequestFeedback
func (RequestFeedback) TableName() string {
	return "request_feedback"
}

// PersistenceEvent represents events that can be processed asynchronously
type PersistenceEvent[T any] struct {
	Type EventType `json:"type"`
	Data T         `json:"data"`
}

// EventType represents the type of persistence event
type EventType string

const (
	EventTypeCreateRequest  EventType = "create_request"
	EventTypeUpdateRequest  EventType = "update_request"
	EventTypeCreateMetrics  EventType = "create_metrics"
	EventTypeCreateFeedback EventType = "create_feedback"
)

// CreateRequestEvent data for creating a new request record
type CreateRequestEvent struct {
	RequestID   uuid.UUID       `json:"request_id"`
	RequestData json.RawMessage `json:"request_data"`
	Model       string          `json:"model"`
	IsStreaming bool            `json:"is_streaming"`
}

// UpdateRequestEvent data for updating request with response
type UpdateRequestEvent struct {
	RequestID    uuid.UUID       `json:"request_id"`
	ResponseData json.RawMessage `json:"response_data"`
	Status       RequestStatus   `json:"status"`
}

// CreateMetricsEvent data for creating request metrics
type CreateMetricsEvent struct {
	RequestID  uuid.UUID `json:"request_id"`
	TotalCost  float64   `json:"total_cost"`
	TokensUsed int       `json:"tokens_used"`
	LatencyMs  int64     `json:"latency_ms"`
}

// CreateFeedbackEvent data for creating request feedback
type CreateFeedbackEvent struct {
	RequestID    uuid.UUID `json:"request_id"`
	FeedbackText string    `json:"feedback_text"`
	Score        float64   `json:"score"`
}
