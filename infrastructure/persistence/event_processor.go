package persistence

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"llm-router/domain/persistence"

	"github.com/google/uuid"
	"github.com/pgvector/pgvector-go"
	"github.com/sirupsen/logrus"
)

// EventProcessor implements persistence.EventProcessor
type EventProcessor struct {
	requestRepo  persistence.RequestRepository
	metricsRepo  persistence.MetricsRepository
	feedbackRepo persistence.FeedbackRepository
	eventChan    chan any
	workerCount  int
	bufferSize   int

	// State management
	ctx            context.Context
	cancel         context.CancelFunc
	wg             sync.WaitGroup
	isRunning      atomic.Bool
	processedCount atomic.Int64
	errorCount     atomic.Int64

	// Health monitoring
	lastProcessedTime atomic.Value
}

// NewEventProcessor creates a new event processor
func NewEventProcessor(
	requestRepo persistence.RequestRepository,
	metricsRepo persistence.MetricsRepository,
	feedbackRepo persistence.FeedbackRepository,
	workerCount int,
	bufferSize int,
) *EventProcessor {
	if workerCount <= 0 {
		workerCount = 5 // Default worker count
	}
	if bufferSize <= 0 {
		bufferSize = 1000 // Default buffer size
	}

	return &EventProcessor{
		requestRepo:  requestRepo,
		metricsRepo:  metricsRepo,
		feedbackRepo: feedbackRepo,
		eventChan:    make(chan interface{}, bufferSize),
		workerCount:  workerCount,
		bufferSize:   bufferSize,
	}
}

// Start begins processing events from the channel
func (ep *EventProcessor) Start(ctx context.Context) error {
	if ep.isRunning.Load() {
		return fmt.Errorf("event processor is already running")
	}

	ep.ctx, ep.cancel = context.WithCancel(ctx)
	ep.isRunning.Store(true)
	ep.lastProcessedTime.Store(time.Now())

	// Start worker goroutines
	for i := 0; i < ep.workerCount; i++ {
		ep.wg.Add(1)
		go ep.worker(i)
	}

	logrus.WithFields(logrus.Fields{
		"worker_count": ep.workerCount,
		"buffer_size":  ep.bufferSize,
	}).Info("Event processor started")

	return nil
}

// Stop gracefully shuts down the event processor
func (ep *EventProcessor) Stop() error {
	if !ep.isRunning.Load() {
		return nil
	}

	logrus.Info("Stopping event processor...")

	// Cancel context to signal workers to stop
	ep.cancel()

	// Close the event channel to prevent new events
	close(ep.eventChan)

	// Wait for all workers to finish with timeout
	done := make(chan struct{})
	go func() {
		ep.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		logrus.Info("Event processor stopped gracefully")
	case <-time.After(30 * time.Second):
		logrus.Warn("Event processor stop timed out")
	}

	ep.isRunning.Store(false)
	return nil
}

// ProcessEvent sends an event to be processed asynchronously
func (ep *EventProcessor) ProcessEvent(event interface{}) error {
	if !ep.isRunning.Load() {
		return fmt.Errorf("event processor is not running")
	}

	select {
	case ep.eventChan <- event:
		return nil
	case <-ep.ctx.Done():
		return fmt.Errorf("event processor is shutting down")
	default:
		// Channel is full, increment error count but don't block
		ep.errorCount.Add(1)
		logrus.Warn("Event processor queue is full, dropping event")
		return fmt.Errorf("event processor queue is full")
	}
}

// Health returns the health status of the processor
func (ep *EventProcessor) Health() persistence.ProcessorHealth {
	queueSize := len(ep.eventChan)

	return persistence.ProcessorHealth{
		IsRunning:      ep.isRunning.Load(),
		QueueSize:      queueSize,
		ProcessedCount: ep.processedCount.Load(),
		ErrorCount:     ep.errorCount.Load(),
	}
}

// worker processes events from the channel
func (ep *EventProcessor) worker(workerID int) {
	defer ep.wg.Done()

	logger := logrus.WithField("worker_id", workerID)
	logger.Info("Event processor worker started")

	for {
		select {
		case event, ok := <-ep.eventChan:
			if !ok {
				logger.Info("Event channel closed, worker stopping")
				return
			}

			// Use processor context and add a per-op timeout to avoid long hangs
			opCtx, cancel := context.WithTimeout(ep.ctx, 10*time.Second)
			if err := ep.processEvent(opCtx, event); err != nil {
				ep.errorCount.Add(1)
				logger.WithError(err).Error("Failed to process event")
			} else {
				ep.processedCount.Add(1)
				ep.lastProcessedTime.Store(time.Now())
			}
			cancel()

		case <-ep.ctx.Done():
			logger.Info("Context cancelled, worker stopping")
			return
		}
	}
}

// processEvent handles individual events
func (ep *EventProcessor) processEvent(ctx context.Context, event interface{}) error {
	switch e := event.(type) {
	case persistence.PersistenceEvent[persistence.CreateRequestEvent]:
		return ep.handleCreateRequest(ctx, e.Data)

	case persistence.PersistenceEvent[persistence.UpdateRequestEvent]:
		return ep.handleUpdateRequest(ctx, e.Data)

	case persistence.PersistenceEvent[persistence.CreateMetricsEvent]:
		return ep.handleCreateMetrics(ctx, e.Data)

	case persistence.PersistenceEvent[persistence.CreateFeedbackEvent]:
		return ep.handleCreateFeedback(ctx, e.Data)

	// Handle direct event types for convenience
	case persistence.CreateRequestEvent:
		return ep.handleCreateRequest(ctx, e)

	case persistence.UpdateRequestEvent:
		return ep.handleUpdateRequest(ctx, e)

	case persistence.CreateMetricsEvent:
		return ep.handleCreateMetrics(ctx, e)

	case persistence.CreateFeedbackEvent:
		return ep.handleCreateFeedback(ctx, e)

	default:
		return fmt.Errorf("unknown event type: %T", event)
	}
}

// handleCreateRequest creates a new request record with proper error handling
func (ep *EventProcessor) handleCreateRequest(ctx context.Context, event persistence.CreateRequestEvent) error {
	record := &persistence.RequestRecord{
		ID:          event.RequestID,
		RequestData: event.RequestData,
		Model:       event.Model,
		IsStreaming: event.IsStreaming,
		Status:      persistence.RequestStatusPending,
		Embedding:   nil, // Will be set later when embedding is generated
	}

	// Attempt to create the record with retry logic for transient errors
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		if err := ep.requestRepo.Create(ctx, record); err != nil {
			lastErr = err

			// Check if this is a vector dimension error (transient)
			if strings.Contains(err.Error(), "vector must have at least 1 dimension") {
				logrus.WithError(err).WithFields(logrus.Fields{
					"request_id": event.RequestID,
					"attempt":    attempt + 1,
				}).Warn("Vector dimension error during request creation, retrying...")

				// Ensure embedding is nil for retry
				record.Embedding = nil

				// Wait before retry
				time.Sleep(time.Duration(attempt+1) * 100 * time.Millisecond)
				continue
			}

			// For non-transient errors, fail immediately
			return fmt.Errorf("failed to create request record: %w", err)
		}

		// Success
		return nil
	}

	// All retries failed
	return fmt.Errorf("failed to create request record after 3 attempts: %w", lastErr)
}

// handleUpdateRequest updates an existing request record
func (ep *EventProcessor) handleUpdateRequest(ctx context.Context, event persistence.UpdateRequestEvent) error {
	// Find the existing record
	record, err := ep.requestRepo.FindByID(ctx, event.RequestID)
	if err != nil {
		return fmt.Errorf("failed to find request for update: %w", err)
	}

	// Update fields
	record.ResponseData = event.ResponseData
	record.Status = event.Status

	return ep.requestRepo.Update(ctx, record)
}

// UpdateRequestEmbedding updates the embedding for an existing request
func (ep *EventProcessor) UpdateRequestEmbedding(ctx context.Context, requestID uuid.UUID, embedding []float32) error {
	// Find the existing record
	record, err := ep.requestRepo.FindByID(ctx, requestID)
	if err != nil {
		return fmt.Errorf("failed to find request for embedding update: %w", err)
	}

	// Convert embedding to pgvector format - ensure we never set empty vectors
	if len(embedding) > 0 {
		vector := pgvector.NewVector(embedding)
		record.Embedding = &vector
	} else {
		// Explicitly set to nil to avoid any GORM serialization issues
		record.Embedding = nil
	}

	return ep.requestRepo.Update(ctx, record)
}

// handleCreateMetrics creates or updates request metrics with proper validation
func (ep *EventProcessor) handleCreateMetrics(ctx context.Context, event persistence.CreateMetricsEvent) error {
	// First check if the request exists with retry logic
	var err error

	for attempt := 0; attempt < 3; attempt++ {
		_, err = ep.requestRepo.FindByID(ctx, event.RequestID)
		if err == nil {
			break // Success
		}

		// Check if this is a "record not found" error
		if strings.Contains(err.Error(), "record not found") {
			logrus.WithError(err).WithFields(logrus.Fields{
				"request_id": event.RequestID,
				"attempt":    attempt + 1,
			}).Warn("Request not found for metrics creation, retrying...")

			// Wait before retry (request might still be being created)
			time.Sleep(time.Duration(attempt+1) * 200 * time.Millisecond)
			continue
		}

		// For other errors, fail immediately
		return fmt.Errorf("failed to find request for metrics: %w", err)
	}

	if err != nil {
		logrus.WithError(err).WithField("request_id", event.RequestID).Warn("Cannot create metrics: request not found after retries")
		return fmt.Errorf("cannot create metrics for non-existent request: %w", err)
	}

	metrics := &persistence.RequestMetrics{
		RequestID:  event.RequestID,
		TotalCost:  event.TotalCost,
		TokensUsed: event.TokensUsed,
		LatencyMs:  event.LatencyMs,
	}

	return ep.metricsRepo.CreateOrUpdate(ctx, metrics)
}

// handleCreateFeedback creates new request feedback with proper validation
func (ep *EventProcessor) handleCreateFeedback(ctx context.Context, event persistence.CreateFeedbackEvent) error {
	// First check if the request exists with retry logic
	var err error

	for attempt := 0; attempt < 3; attempt++ {
		_, err = ep.requestRepo.FindByID(ctx, event.RequestID)
		if err == nil {
			break // Success
		}

		// Check if this is a "record not found" error
		if strings.Contains(err.Error(), "record not found") {
			logrus.WithError(err).WithFields(logrus.Fields{
				"request_id": event.RequestID,
				"attempt":    attempt + 1,
			}).Warn("Request not found for feedback creation, retrying...")

			// Wait before retry (request might still be being created)
			time.Sleep(time.Duration(attempt+1) * 200 * time.Millisecond)
			continue
		}

		// For other errors, fail immediately
		return fmt.Errorf("failed to find request for feedback: %w", err)
	}

	if err != nil {
		logrus.WithError(err).WithField("request_id", event.RequestID).Warn("Cannot create feedback: request not found after retries")
		return fmt.Errorf("cannot create feedback for non-existent request: %w", err)
	}

	feedback := &persistence.RequestFeedback{
		RequestID:    event.RequestID,
		FeedbackText: event.FeedbackText,
		Score:        event.Score,
	}

	return ep.feedbackRepo.Create(ctx, feedback)
}

// RequestTracker implements persistence.RequestTracker using the event processor
type RequestTracker struct {
	processor persistence.EventProcessor
}

// NewRequestTracker creates a new request tracker
func NewRequestTracker(processor persistence.EventProcessor) persistence.RequestTracker {
	return &RequestTracker{
		processor: processor,
	}
}

// StartTracking begins tracking a new request
func (rt *RequestTracker) StartTracking(ctx context.Context, requestID uuid.UUID, requestData []byte, model string, isStreaming bool) error {
	event := persistence.CreateRequestEvent{
		RequestID:   requestID,
		RequestData: json.RawMessage(requestData),
		Model:       model,
		IsStreaming: isStreaming,
	}

	return rt.processor.ProcessEvent(event)
}

// CompleteTracking finalizes request tracking with response data
func (rt *RequestTracker) CompleteTracking(ctx context.Context, requestID uuid.UUID, responseData []byte, metrics persistence.RequestMetrics) error {
	// Update request with response data
	updateEvent := persistence.UpdateRequestEvent{
		RequestID:    requestID,
		ResponseData: json.RawMessage(responseData),
		Status:       persistence.RequestStatusCompleted,
	}

	if err := rt.processor.ProcessEvent(updateEvent); err != nil {
		return fmt.Errorf("failed to process update request event: %w", err)
	}

	// Create metrics record
	metricsEvent := persistence.CreateMetricsEvent{
		RequestID:  requestID,
		TotalCost:  metrics.TotalCost,
		TokensUsed: metrics.TokensUsed,
		LatencyMs:  metrics.LatencyMs,
	}

	if err := rt.processor.ProcessEvent(metricsEvent); err != nil {
		return fmt.Errorf("failed to process create metrics event: %w", err)
	}

	return nil
}

// FailTracking marks a request as failed
func (rt *RequestTracker) FailTracking(ctx context.Context, requestID uuid.UUID, errorMsg string) error {
	// Create error response data
	errorData, _ := json.Marshal(map[string]string{
		"error": errorMsg,
	})

	event := persistence.UpdateRequestEvent{
		RequestID:    requestID,
		ResponseData: json.RawMessage(errorData),
		Status:       persistence.RequestStatusFailed,
	}

	return rt.processor.ProcessEvent(event)
}

// SubmitFeedback adds feedback for a request
func (rt *RequestTracker) SubmitFeedback(ctx context.Context, requestID uuid.UUID, feedbackText string, score float64) error {
	event := persistence.CreateFeedbackEvent{
		RequestID:    requestID,
		FeedbackText: feedbackText,
		Score:        score,
	}

	return rt.processor.ProcessEvent(event)
}

// UpdateEmbedding updates the embedding for a request
func (rt *RequestTracker) UpdateEmbedding(ctx context.Context, requestID uuid.UUID, embedding []float32) error {
	// Cast to concrete type to access UpdateRequestEmbedding method
	if ep, ok := rt.processor.(*EventProcessor); ok {
		return ep.UpdateRequestEmbedding(ctx, requestID, embedding)
	}
	return fmt.Errorf("processor does not support embedding updates")
}
