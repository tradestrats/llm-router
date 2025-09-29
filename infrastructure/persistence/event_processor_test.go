package persistence

import (
	"context"
	"fmt"
	"testing"
	"time"

	"llm-router/domain/persistence"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// Mock repositories
type MockRequestRepository struct {
	mock.Mock
}

func (m *MockRequestRepository) Create(ctx context.Context, entity *persistence.RequestRecord) error {
	args := m.Called(ctx, entity)
	return args.Error(0)
}

func (m *MockRequestRepository) Update(ctx context.Context, entity *persistence.RequestRecord) error {
	args := m.Called(ctx, entity)
	return args.Error(0)
}

func (m *MockRequestRepository) FindByID(ctx context.Context, id uuid.UUID) (*persistence.RequestRecord, error) {
	args := m.Called(ctx, id)
	return args.Get(0).(*persistence.RequestRecord), args.Error(1)
}

func (m *MockRequestRepository) Delete(ctx context.Context, id uuid.UUID) error {
	args := m.Called(ctx, id)
	return args.Error(0)
}

func (m *MockRequestRepository) FindByIDWithRelations(ctx context.Context, id uuid.UUID) (*persistence.RequestRecord, error) {
	args := m.Called(ctx, id)
	return args.Get(0).(*persistence.RequestRecord), args.Error(1)
}

func (m *MockRequestRepository) FindByStatus(ctx context.Context, status persistence.RequestStatus, limit int) ([]*persistence.RequestRecord, error) {
	args := m.Called(ctx, status, limit)
	return args.Get(0).([]*persistence.RequestRecord), args.Error(1)
}

func (m *MockRequestRepository) FindRecent(ctx context.Context, limit int) ([]*persistence.RequestRecord, error) {
	args := m.Called(ctx, limit)
	return args.Get(0).([]*persistence.RequestRecord), args.Error(1)
}

func (m *MockRequestRepository) UpdateStatus(ctx context.Context, id uuid.UUID, status persistence.RequestStatus) error {
	args := m.Called(ctx, id, status)
	return args.Error(0)
}

type MockMetricsRepository struct {
	mock.Mock
}

func (m *MockMetricsRepository) Create(ctx context.Context, entity *persistence.RequestMetrics) error {
	args := m.Called(ctx, entity)
	return args.Error(0)
}

func (m *MockMetricsRepository) Update(ctx context.Context, entity *persistence.RequestMetrics) error {
	args := m.Called(ctx, entity)
	return args.Error(0)
}

func (m *MockMetricsRepository) FindByID(ctx context.Context, id uuid.UUID) (*persistence.RequestMetrics, error) {
	args := m.Called(ctx, id)
	return args.Get(0).(*persistence.RequestMetrics), args.Error(1)
}

func (m *MockMetricsRepository) Delete(ctx context.Context, id uuid.UUID) error {
	args := m.Called(ctx, id)
	return args.Error(0)
}

func (m *MockMetricsRepository) FindByRequestID(ctx context.Context, requestID uuid.UUID) (*persistence.RequestMetrics, error) {
	args := m.Called(ctx, requestID)
	return args.Get(0).(*persistence.RequestMetrics), args.Error(1)
}

func (m *MockMetricsRepository) CreateOrUpdate(ctx context.Context, metrics *persistence.RequestMetrics) error {
	args := m.Called(ctx, metrics)
	return args.Error(0)
}

func (m *MockMetricsRepository) GetAggregatedMetrics(ctx context.Context, limit int) (*persistence.AggregatedMetrics, error) {
	args := m.Called(ctx, limit)
	return args.Get(0).(*persistence.AggregatedMetrics), args.Error(1)
}

type MockFeedbackRepository struct {
	mock.Mock
}

func (m *MockFeedbackRepository) Create(ctx context.Context, entity *persistence.RequestFeedback) error {
	args := m.Called(ctx, entity)
	return args.Error(0)
}

func (m *MockFeedbackRepository) Update(ctx context.Context, entity *persistence.RequestFeedback) error {
	args := m.Called(ctx, entity)
	return args.Error(0)
}

func (m *MockFeedbackRepository) FindByID(ctx context.Context, id uuid.UUID) (*persistence.RequestFeedback, error) {
	args := m.Called(ctx, id)
	return args.Get(0).(*persistence.RequestFeedback), args.Error(1)
}

func (m *MockFeedbackRepository) Delete(ctx context.Context, id uuid.UUID) error {
	args := m.Called(ctx, id)
	return args.Error(0)
}

func (m *MockFeedbackRepository) FindByRequestID(ctx context.Context, requestID uuid.UUID) ([]*persistence.RequestFeedback, error) {
	args := m.Called(ctx, requestID)
	return args.Get(0).([]*persistence.RequestFeedback), args.Error(1)
}

func (m *MockFeedbackRepository) GetAverageScore(ctx context.Context, requestID *uuid.UUID) (float64, error) {
	args := m.Called(ctx, requestID)
	return args.Get(0).(float64), args.Error(1)
}

func (m *MockFeedbackRepository) FindRecentFeedback(ctx context.Context, limit int) ([]*persistence.RequestFeedback, error) {
	args := m.Called(ctx, limit)
	return args.Get(0).([]*persistence.RequestFeedback), args.Error(1)
}

func TestEventProcessor_StartStop(t *testing.T) {
	requestRepo := &MockRequestRepository{}
	metricsRepo := &MockMetricsRepository{}
	feedbackRepo := &MockFeedbackRepository{}

	processor := NewEventProcessor(requestRepo, metricsRepo, feedbackRepo, 2, 10)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Test start
	err := processor.Start(ctx)
	assert.NoError(t, err)

	// Check health
	health := processor.Health()
	assert.True(t, health.IsRunning)
	assert.Equal(t, 0, health.QueueSize)

	// Test duplicate start (should fail)
	err = processor.Start(ctx)
	assert.Error(t, err)

	// Test stop
	err = processor.Stop()
	assert.NoError(t, err)

	// Check health after stop
	health = processor.Health()
	assert.False(t, health.IsRunning)
}

func TestEventProcessor_ProcessCreateRequestEvent(t *testing.T) {
	requestRepo := &MockRequestRepository{}
	metricsRepo := &MockMetricsRepository{}
	feedbackRepo := &MockFeedbackRepository{}

	processor := NewEventProcessor(requestRepo, metricsRepo, feedbackRepo, 1, 10)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start processor
	err := processor.Start(ctx)
	assert.NoError(t, err)
	defer processor.Stop()

	// Setup mock expectations
	requestRepo.On("Create", mock.Anything, mock.AnythingOfType("*persistence.RequestRecord")).Return(nil)

	// Create test event
	requestID := uuid.New()
	event := persistence.CreateRequestEvent{
		RequestID:   requestID,
		RequestData: []byte(`{"messages":[{"role":"user","content":"test"}]}`),
		Model:       "test-model",
		IsStreaming: false,
	}

	// Process event
	err = processor.ProcessEvent(event)
	assert.NoError(t, err)

	// Wait for processing
	time.Sleep(100 * time.Millisecond)

	// Verify expectations
	requestRepo.AssertExpectations(t)
}

func TestEventProcessor_ProcessCreateMetricsEvent(t *testing.T) {
	requestRepo := &MockRequestRepository{}
	metricsRepo := &MockMetricsRepository{}
	feedbackRepo := &MockFeedbackRepository{}

	processor := NewEventProcessor(requestRepo, metricsRepo, feedbackRepo, 1, 10)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start processor
	err := processor.Start(ctx)
	assert.NoError(t, err)
	defer processor.Stop()

	// Setup mock expectations - need to mock FindByID for the new retry logic
	requestRecord := &persistence.RequestRecord{ID: uuid.New()}
	requestRepo.On("FindByID", mock.Anything, mock.AnythingOfType("uuid.UUID")).Return(requestRecord, nil)
	metricsRepo.On("CreateOrUpdate", mock.Anything, mock.AnythingOfType("*persistence.RequestMetrics")).Return(nil)

	// Create test event
	requestID := uuid.New()
	event := persistence.CreateMetricsEvent{
		RequestID:  requestID,
		TotalCost:  0.05,
		TokensUsed: 1000,
		LatencyMs:  500,
	}

	// Process event
	err = processor.ProcessEvent(event)
	assert.NoError(t, err)

	// Wait for processing
	time.Sleep(100 * time.Millisecond)

	// Verify expectations
	requestRepo.AssertExpectations(t)
	metricsRepo.AssertExpectations(t)
}

func TestEventProcessor_ProcessCreateFeedbackEvent(t *testing.T) {
	requestRepo := &MockRequestRepository{}
	metricsRepo := &MockMetricsRepository{}
	feedbackRepo := &MockFeedbackRepository{}

	processor := NewEventProcessor(requestRepo, metricsRepo, feedbackRepo, 1, 10)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start processor
	err := processor.Start(ctx)
	assert.NoError(t, err)
	defer processor.Stop()

	// Setup mock expectations - need to mock FindByID for the new retry logic
	requestRecord := &persistence.RequestRecord{ID: uuid.New()}
	requestRepo.On("FindByID", mock.Anything, mock.AnythingOfType("uuid.UUID")).Return(requestRecord, nil)
	feedbackRepo.On("Create", mock.Anything, mock.AnythingOfType("*persistence.RequestFeedback")).Return(nil)

	// Create test event
	requestID := uuid.New()
	event := persistence.CreateFeedbackEvent{
		RequestID:    requestID,
		FeedbackText: "Great response!",
		Score:        0.9,
	}

	// Process event
	err = processor.ProcessEvent(event)
	assert.NoError(t, err)

	// Wait for processing
	time.Sleep(100 * time.Millisecond)

	// Verify expectations
	requestRepo.AssertExpectations(t)
	feedbackRepo.AssertExpectations(t)
}

func TestRequestTracker_StartTracking(t *testing.T) {
	requestRepo := &MockRequestRepository{}
	metricsRepo := &MockMetricsRepository{}
	feedbackRepo := &MockFeedbackRepository{}

	processor := NewEventProcessor(requestRepo, metricsRepo, feedbackRepo, 1, 10)
	tracker := NewRequestTracker(processor)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start processor
	err := processor.Start(ctx)
	assert.NoError(t, err)
	defer processor.Stop()

	// Setup mock expectations
	requestRepo.On("Create", mock.Anything, mock.AnythingOfType("*persistence.RequestRecord")).Return(nil)

	// Test start tracking
	requestID := uuid.New()
	requestData := []byte(`{"messages":[{"role":"user","content":"test"}]}`)

	err = tracker.StartTracking(ctx, requestID, requestData, "test-model", false)
	assert.NoError(t, err)

	// Wait for processing
	time.Sleep(100 * time.Millisecond)

	// Verify expectations
	requestRepo.AssertExpectations(t)
}

func TestRequestTracker_CompleteTracking(t *testing.T) {
	requestRepo := &MockRequestRepository{}
	metricsRepo := &MockMetricsRepository{}
	feedbackRepo := &MockFeedbackRepository{}

	processor := NewEventProcessor(requestRepo, metricsRepo, feedbackRepo, 1, 10)
	tracker := NewRequestTracker(processor)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start processor
	err := processor.Start(ctx)
	assert.NoError(t, err)
	defer processor.Stop()

	// Setup mock expectations
	requestRecord := &persistence.RequestRecord{ID: uuid.New()}
	requestRepo.On("FindByID", mock.Anything, mock.AnythingOfType("uuid.UUID")).Return(requestRecord, nil)
	requestRepo.On("Update", mock.Anything, mock.AnythingOfType("*persistence.RequestRecord")).Return(nil)
	metricsRepo.On("CreateOrUpdate", mock.Anything, mock.AnythingOfType("*persistence.RequestMetrics")).Return(nil)

	// Test complete tracking
	requestID := uuid.New()
	responseData := []byte(`{"choices":[{"message":{"content":"test response"}}]}`)
	metrics := persistence.RequestMetrics{
		TotalCost:  0.05,
		TokensUsed: 1000,
		LatencyMs:  500,
	}

	err = tracker.CompleteTracking(ctx, requestID, responseData, metrics)
	assert.NoError(t, err)

	// Wait for processing
	time.Sleep(100 * time.Millisecond)

	// Verify expectations
	requestRepo.AssertExpectations(t)
	metricsRepo.AssertExpectations(t)
}

func TestEventProcessor_QueueFull(t *testing.T) {
	requestRepo := &MockRequestRepository{}
	metricsRepo := &MockMetricsRepository{}
	feedbackRepo := &MockFeedbackRepository{}

	// Create processor with small buffer
	processor := NewEventProcessor(requestRepo, metricsRepo, feedbackRepo, 1, 1)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Setup mock expectations for the first event that will be processed
	requestRepo.On("Create", mock.Anything, mock.AnythingOfType("*persistence.RequestRecord")).Return(nil)

	// Start processor
	err := processor.Start(ctx)
	assert.NoError(t, err)
	defer processor.Stop()

	// Fill the queue
	event := persistence.CreateRequestEvent{
		RequestID:   uuid.New(),
		RequestData: []byte("test"),
		Model:       "test-model",
		IsStreaming: false,
	}

	// First event should succeed
	err = processor.ProcessEvent(event)
	assert.NoError(t, err)

	// Second event should fail (queue full)
	err = processor.ProcessEvent(event)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "queue is full")

	// Verify expectations
	requestRepo.AssertExpectations(t)
}

func TestEventProcessor_HandleCreateRequestWithRetry(t *testing.T) {
	requestRepo := &MockRequestRepository{}
	metricsRepo := &MockMetricsRepository{}
	feedbackRepo := &MockFeedbackRepository{}

	processor := NewEventProcessor(requestRepo, metricsRepo, feedbackRepo, 1, 10)

	ctx := context.Background()

	// Test retry logic for vector dimension error
	requestID := uuid.New()
	event := persistence.CreateRequestEvent{
		RequestID:   requestID,
		RequestData: []byte(`{"messages":[{"role":"user","content":"test"}]}`),
		Model:       "test-model",
		IsStreaming: false,
	}

	// First two calls fail with vector error, third succeeds
	vectorError := fmt.Errorf("failed to create request record: ERROR: vector must have at least 1 dimension (SQLSTATE 22000)")
	requestRepo.On("Create", mock.Anything, mock.AnythingOfType("*persistence.RequestRecord")).Return(vectorError).Twice()
	requestRepo.On("Create", mock.Anything, mock.AnythingOfType("*persistence.RequestRecord")).Return(nil).Once()

	err := processor.handleCreateRequest(ctx, event)
	assert.NoError(t, err)

	// Verify all calls were made
	requestRepo.AssertExpectations(t)
}

func TestEventProcessor_HandleCreateMetricsWithRetry(t *testing.T) {
	requestRepo := &MockRequestRepository{}
	metricsRepo := &MockMetricsRepository{}
	feedbackRepo := &MockFeedbackRepository{}

	processor := NewEventProcessor(requestRepo, metricsRepo, feedbackRepo, 1, 10)

	ctx := context.Background()

	// Test retry logic for "record not found" error
	requestID := uuid.New()
	event := persistence.CreateMetricsEvent{
		RequestID:  requestID,
		TotalCost:  0.05,
		TokensUsed: 1000,
		LatencyMs:  500,
	}

	// First two calls fail with "record not found", third succeeds
	recordNotFoundError := fmt.Errorf("request record not found: record not found")
	requestRecord := &persistence.RequestRecord{ID: requestID}

	requestRepo.On("FindByID", mock.Anything, requestID).Return((*persistence.RequestRecord)(nil), recordNotFoundError).Twice()
	requestRepo.On("FindByID", mock.Anything, requestID).Return(requestRecord, nil).Once()
	metricsRepo.On("CreateOrUpdate", mock.Anything, mock.AnythingOfType("*persistence.RequestMetrics")).Return(nil).Once()

	err := processor.handleCreateMetrics(ctx, event)
	assert.NoError(t, err)

	// Verify all calls were made
	requestRepo.AssertExpectations(t)
	metricsRepo.AssertExpectations(t)
}

func TestEventProcessor_HandleCreateFeedbackWithRetry(t *testing.T) {
	requestRepo := &MockRequestRepository{}
	metricsRepo := &MockMetricsRepository{}
	feedbackRepo := &MockFeedbackRepository{}

	processor := NewEventProcessor(requestRepo, metricsRepo, feedbackRepo, 1, 10)

	ctx := context.Background()

	// Test retry logic for "record not found" error
	requestID := uuid.New()
	event := persistence.CreateFeedbackEvent{
		RequestID:    requestID,
		FeedbackText: "Great response!",
		Score:        0.9,
	}

	// First two calls fail with "record not found", third succeeds
	recordNotFoundError := fmt.Errorf("request record not found: record not found")
	requestRecord := &persistence.RequestRecord{ID: requestID}

	requestRepo.On("FindByID", mock.Anything, requestID).Return((*persistence.RequestRecord)(nil), recordNotFoundError).Twice()
	requestRepo.On("FindByID", mock.Anything, requestID).Return(requestRecord, nil).Once()
	feedbackRepo.On("Create", mock.Anything, mock.AnythingOfType("*persistence.RequestFeedback")).Return(nil).Once()

	err := processor.handleCreateFeedback(ctx, event)
	assert.NoError(t, err)

	// Verify all calls were made
	requestRepo.AssertExpectations(t)
	feedbackRepo.AssertExpectations(t)
}
