package persistence

import (
	"testing"

	"github.com/google/uuid"
	"github.com/pgvector/pgvector-go"
	"github.com/stretchr/testify/assert"
	"gorm.io/gorm"
)

func TestRequestRecord_TableName(t *testing.T) {
	record := RequestRecord{}
	assert.Equal(t, "requests", record.TableName())
}

func TestRequestMetrics_TableName(t *testing.T) {
	metrics := RequestMetrics{}
	assert.Equal(t, "request_metrics", metrics.TableName())
}

func TestRequestFeedback_TableName(t *testing.T) {
	feedback := RequestFeedback{}
	assert.Equal(t, "request_feedback", feedback.TableName())
}

func TestRequestStatus_Constants(t *testing.T) {
	assert.Equal(t, RequestStatus("pending"), RequestStatusPending)
	assert.Equal(t, RequestStatus("completed"), RequestStatusCompleted)
	assert.Equal(t, RequestStatus("failed"), RequestStatusFailed)
}

func TestEventType_Constants(t *testing.T) {
	assert.Equal(t, EventType("create_request"), EventTypeCreateRequest)
	assert.Equal(t, EventType("update_request"), EventTypeUpdateRequest)
	assert.Equal(t, EventType("create_metrics"), EventTypeCreateMetrics)
	assert.Equal(t, EventType("create_feedback"), EventTypeCreateFeedback)
}

func TestPersistenceEvent_GenericType(t *testing.T) {
	// Test CreateRequestEvent
	requestEvent := PersistenceEvent[CreateRequestEvent]{
		Type: EventTypeCreateRequest,
		Data: CreateRequestEvent{
			RequestID:   uuid.New(),
			RequestData: []byte(`{"test": "data"}`),
			Model:       "test-model",
			IsStreaming: false,
		},
	}

	assert.Equal(t, EventTypeCreateRequest, requestEvent.Type)
	assert.NotEmpty(t, requestEvent.Data.RequestID)
	assert.Equal(t, "test-model", requestEvent.Data.Model)
	assert.False(t, requestEvent.Data.IsStreaming)

	// Test CreateMetricsEvent
	metricsEvent := PersistenceEvent[CreateMetricsEvent]{
		Type: EventTypeCreateMetrics,
		Data: CreateMetricsEvent{
			RequestID:  uuid.New(),
			TotalCost:  1.25,
			TokensUsed: 1000,
			LatencyMs:  500,
		},
	}

	assert.Equal(t, EventTypeCreateMetrics, metricsEvent.Type)
	assert.Equal(t, 1.25, metricsEvent.Data.TotalCost)
	assert.Equal(t, 1000, metricsEvent.Data.TokensUsed)
	assert.Equal(t, int64(500), metricsEvent.Data.LatencyMs)

	// Test CreateFeedbackEvent
	feedbackEvent := PersistenceEvent[CreateFeedbackEvent]{
		Type: EventTypeCreateFeedback,
		Data: CreateFeedbackEvent{
			RequestID:    uuid.New(),
			FeedbackText: "Great response!",
			Score:        0.9,
		},
	}

	assert.Equal(t, EventTypeCreateFeedback, feedbackEvent.Type)
	assert.Equal(t, "Great response!", feedbackEvent.Data.FeedbackText)
	assert.Equal(t, 0.9, feedbackEvent.Data.Score)
}

func TestRequestRecord_UUIDGeneration(t *testing.T) {
	record := RequestRecord{}

	// Initially, ID should be nil UUID
	assert.Equal(t, uuid.Nil, record.ID)

	// After setting a UUID, it should be different
	record.ID = uuid.New()
	assert.NotEqual(t, uuid.Nil, record.ID)
}

func TestRequestMetrics_Validation(t *testing.T) {
	requestID := uuid.New()
	metrics := RequestMetrics{
		RequestID:  requestID,
		TotalCost:  5.50,
		TokensUsed: 2000,
		LatencyMs:  750,
	}

	assert.Equal(t, requestID, metrics.RequestID)
	assert.Equal(t, 5.50, metrics.TotalCost)
	assert.Equal(t, 2000, metrics.TokensUsed)
	assert.Equal(t, int64(750), metrics.LatencyMs)
}

func TestRequestFeedback_ScoreValidation(t *testing.T) {
	requestID := uuid.New()

	// Test valid scores
	validScores := []float64{0.0, 0.5, 1.0, 0.25, 0.75}
	for _, score := range validScores {
		feedback := RequestFeedback{
			RequestID:    requestID,
			FeedbackText: "Test feedback",
			Score:        score,
		}

		assert.Equal(t, score, feedback.Score)
		assert.True(t, feedback.Score >= 0.0 && feedback.Score <= 1.0)
	}
}

func TestRequestRecord_BeforeCreate(t *testing.T) {
	tests := []struct {
		name     string
		record   *RequestRecord
		expected *RequestRecord
	}{
		{
			name: "nil embedding remains nil",
			record: &RequestRecord{
				ID:        uuid.Nil,
				Embedding: nil,
			},
			expected: &RequestRecord{
				ID:        uuid.New(), // Will be set by BeforeCreate
				Embedding: nil,
			},
		},
		{
			name: "empty embedding vector becomes nil",
			record: &RequestRecord{
				ID:        uuid.Nil,
				Embedding: &pgvector.Vector{}, // Empty vector
			},
			expected: &RequestRecord{
				ID:        uuid.New(), // Will be set by BeforeCreate
				Embedding: nil,        // Should be converted to nil
			},
		},
		{
			name: "valid embedding vector remains unchanged",
			record: &RequestRecord{
				ID:        uuid.Nil,
				Embedding: func() *pgvector.Vector { v := pgvector.NewVector([]float32{0.1, 0.2, 0.3}); return &v }(),
			},
			expected: &RequestRecord{
				ID:        uuid.New(), // Will be set by BeforeCreate
				Embedding: func() *pgvector.Vector { v := pgvector.NewVector([]float32{0.1, 0.2, 0.3}); return &v }(),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a mock GORM DB
			db := &gorm.DB{}

			// Call BeforeCreate
			err := tt.record.BeforeCreate(db)
			assert.NoError(t, err)

			// Check that ID was set
			assert.NotEqual(t, uuid.Nil, tt.record.ID)

			// Check embedding handling
			if tt.expected.Embedding == nil {
				assert.Nil(t, tt.record.Embedding)
			} else {
				assert.NotNil(t, tt.record.Embedding)
				assert.Equal(t, tt.expected.Embedding.Slice(), tt.record.Embedding.Slice())
			}
		})
	}
}

func TestRequestRecord_BeforeUpdate(t *testing.T) {
	tests := []struct {
		name     string
		record   *RequestRecord
		expected *RequestRecord
	}{
		{
			name: "nil embedding remains nil",
			record: &RequestRecord{
				Embedding: nil,
			},
			expected: &RequestRecord{
				Embedding: nil,
			},
		},
		{
			name: "empty embedding vector becomes nil",
			record: &RequestRecord{
				Embedding: &pgvector.Vector{}, // Empty vector
			},
			expected: &RequestRecord{
				Embedding: nil, // Should be converted to nil
			},
		},
		{
			name: "valid embedding vector remains unchanged",
			record: &RequestRecord{
				Embedding: func() *pgvector.Vector { v := pgvector.NewVector([]float32{0.1, 0.2, 0.3}); return &v }(),
			},
			expected: &RequestRecord{
				Embedding: func() *pgvector.Vector { v := pgvector.NewVector([]float32{0.1, 0.2, 0.3}); return &v }(),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a mock GORM DB
			db := &gorm.DB{}

			// Call BeforeUpdate
			err := tt.record.BeforeUpdate(db)
			assert.NoError(t, err)

			// Check embedding handling
			if tt.expected.Embedding == nil {
				assert.Nil(t, tt.record.Embedding)
			} else {
				assert.NotNil(t, tt.record.Embedding)
				assert.Equal(t, tt.expected.Embedding.Slice(), tt.record.Embedding.Slice())
			}
		})
	}
}
