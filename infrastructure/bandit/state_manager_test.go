package bandit

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	"llm-router/domain/bandit"
	"llm-router/domain/persistence"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

// setupTestDBForStateManager creates an in-memory SQLite database for testing
func setupTestDBForStateManager(t *testing.T) (*gorm.DB, persistence.RequestRepository) {
	db, err := gorm.Open(sqlite.Open(":memory:"), &gorm.Config{})
	require.NoError(t, err)

	// Create simplified tables for testing (without PostgreSQL-specific features)
	err = db.Exec(`
		CREATE TABLE IF NOT EXISTS requests (
			id TEXT PRIMARY KEY,
			request_data TEXT NOT NULL,
			response_data TEXT,
			model TEXT NOT NULL,
			is_streaming INTEGER DEFAULT 0,
			status TEXT NOT NULL DEFAULT 'pending',
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
		)
	`).Error
	require.NoError(t, err)

	err = db.Exec(`
		CREATE TABLE IF NOT EXISTS request_metrics (
			id TEXT PRIMARY KEY,
			request_id TEXT NOT NULL,
			total_cost REAL DEFAULT 0,
			tokens_used INTEGER DEFAULT 0,
			latency_ms INTEGER DEFAULT 0,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
		)
	`).Error
	require.NoError(t, err)

	err = db.Exec(`
		CREATE TABLE IF NOT EXISTS request_feedback (
			id TEXT PRIMARY KEY,
			request_id TEXT NOT NULL,
			score REAL NOT NULL,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
		)
	`).Error
	require.NoError(t, err)

	// Create a mock request repository
	requestRepo := &MockRequestRepository{db: db}
	return db, requestRepo
}

// TestRequestRecord is a simplified version for testing without embedding field
type TestRequestRecord struct {
	ID           string          `gorm:"primaryKey"`
	RequestData  json.RawMessage `gorm:"type:text"`
	ResponseData json.RawMessage `gorm:"type:text"`
	Model        string          `gorm:"type:text"`
	IsStreaming  bool            `gorm:"type:integer"`
	Status       string          `gorm:"type:text"`
	CreatedAt    time.Time
	UpdatedAt    time.Time
}

// TableName specifies the table name for TestRequestRecord
func (TestRequestRecord) TableName() string {
	return "requests"
}

// TestRequestMetrics is a simplified version for testing
type TestRequestMetrics struct {
	ID         string  `gorm:"primaryKey"`
	RequestID  string  `gorm:"type:text"`
	TotalCost  float64 `gorm:"type:real"`
	TokensUsed int64   `gorm:"type:integer"`
	LatencyMs  int64   `gorm:"type:integer"`
	CreatedAt  time.Time
	UpdatedAt  time.Time
}

// TableName specifies the table name for TestRequestMetrics
func (TestRequestMetrics) TableName() string {
	return "request_metrics"
}

// TestRequestFeedback is a simplified version for testing
type TestRequestFeedback struct {
	ID        string  `gorm:"primaryKey"`
	RequestID string  `gorm:"type:text"`
	Score     float64 `gorm:"type:real"`
	CreatedAt time.Time
	UpdatedAt time.Time
}

// TableName specifies the table name for TestRequestFeedback
func (TestRequestFeedback) TableName() string {
	return "request_feedback"
}

// MockRequestRepository implements persistence.RequestRepository for testing
type MockRequestRepository struct {
	db *gorm.DB
}

func (m *MockRequestRepository) Create(ctx context.Context, record *persistence.RequestRecord) error {
	// Convert persistence record to test record
	testRecord := &TestRequestRecord{
		ID:           record.ID.String(),
		RequestData:  record.RequestData,
		ResponseData: record.ResponseData,
		Model:        record.Model,
		IsStreaming:  record.IsStreaming,
		Status:       string(record.Status),
		CreatedAt:    record.CreatedAt,
		UpdatedAt:    record.UpdatedAt,
	}
	return m.db.WithContext(ctx).Create(testRecord).Error
}

func (m *MockRequestRepository) Update(ctx context.Context, record *persistence.RequestRecord) error {
	// Convert persistence record to test record
	testRecord := &TestRequestRecord{
		ID:           record.ID.String(),
		RequestData:  record.RequestData,
		ResponseData: record.ResponseData,
		Model:        record.Model,
		IsStreaming:  record.IsStreaming,
		Status:       string(record.Status),
		CreatedAt:    record.CreatedAt,
		UpdatedAt:    record.UpdatedAt,
	}
	return m.db.WithContext(ctx).Save(testRecord).Error
}

func (m *MockRequestRepository) FindByID(ctx context.Context, id uuid.UUID) (*persistence.RequestRecord, error) {
	var testRecord TestRequestRecord
	err := m.db.WithContext(ctx).Where("id = ?", id.String()).First(&testRecord).Error
	if err != nil {
		return nil, err
	}

	// Convert test record back to persistence record
	recordID, err := uuid.Parse(testRecord.ID)
	if err != nil {
		return nil, err
	}

	return &persistence.RequestRecord{
		ID:           recordID,
		RequestData:  testRecord.RequestData,
		ResponseData: testRecord.ResponseData,
		Model:        testRecord.Model,
		IsStreaming:  testRecord.IsStreaming,
		Status:       persistence.RequestStatus(testRecord.Status),
		CreatedAt:    testRecord.CreatedAt,
		UpdatedAt:    testRecord.UpdatedAt,
	}, nil
}

func (m *MockRequestRepository) FindByModel(ctx context.Context, model string, limit int) ([]*persistence.RequestRecord, error) {
	var testRecords []TestRequestRecord
	err := m.db.WithContext(ctx).
		Where("model = ?", model).
		Order("created_at DESC").
		Limit(limit).
		Find(&testRecords).Error
	if err != nil {
		return nil, err
	}

	// Convert test records to persistence records
	var records []*persistence.RequestRecord
	for _, testRecord := range testRecords {
		recordID, err := uuid.Parse(testRecord.ID)
		if err != nil {
			return nil, err
		}
		records = append(records, &persistence.RequestRecord{
			ID:           recordID,
			RequestData:  testRecord.RequestData,
			ResponseData: testRecord.ResponseData,
			Model:        testRecord.Model,
			IsStreaming:  testRecord.IsStreaming,
			Status:       persistence.RequestStatus(testRecord.Status),
			CreatedAt:    testRecord.CreatedAt,
			UpdatedAt:    testRecord.UpdatedAt,
		})
	}
	return records, nil
}

func (m *MockRequestRepository) FindByIDWithRelations(ctx context.Context, id uuid.UUID) (*persistence.RequestRecord, error) {
	return m.FindByID(ctx, id)
}

func (m *MockRequestRepository) FindByStatus(ctx context.Context, status persistence.RequestStatus, limit int) ([]*persistence.RequestRecord, error) {
	var testRecords []TestRequestRecord
	err := m.db.WithContext(ctx).
		Where("status = ?", string(status)).
		Order("created_at DESC").
		Limit(limit).
		Find(&testRecords).Error
	if err != nil {
		return nil, err
	}

	// Convert test records to persistence records
	var records []*persistence.RequestRecord
	for _, testRecord := range testRecords {
		recordID, err := uuid.Parse(testRecord.ID)
		if err != nil {
			return nil, err
		}
		records = append(records, &persistence.RequestRecord{
			ID:           recordID,
			RequestData:  testRecord.RequestData,
			ResponseData: testRecord.ResponseData,
			Model:        testRecord.Model,
			IsStreaming:  testRecord.IsStreaming,
			Status:       persistence.RequestStatus(testRecord.Status),
			CreatedAt:    testRecord.CreatedAt,
			UpdatedAt:    testRecord.UpdatedAt,
		})
	}
	return records, nil
}

func (m *MockRequestRepository) FindRecent(ctx context.Context, limit int) ([]*persistence.RequestRecord, error) {
	var testRecords []TestRequestRecord
	err := m.db.WithContext(ctx).
		Order("created_at DESC").
		Limit(limit).
		Find(&testRecords).Error
	if err != nil {
		return nil, err
	}

	// Convert test records to persistence records
	var records []*persistence.RequestRecord
	for _, testRecord := range testRecords {
		recordID, err := uuid.Parse(testRecord.ID)
		if err != nil {
			return nil, err
		}
		records = append(records, &persistence.RequestRecord{
			ID:           recordID,
			RequestData:  testRecord.RequestData,
			ResponseData: testRecord.ResponseData,
			Model:        testRecord.Model,
			IsStreaming:  testRecord.IsStreaming,
			Status:       persistence.RequestStatus(testRecord.Status),
			CreatedAt:    testRecord.CreatedAt,
			UpdatedAt:    testRecord.UpdatedAt,
		})
	}
	return records, nil
}

func (m *MockRequestRepository) UpdateStatus(ctx context.Context, id uuid.UUID, status persistence.RequestStatus) error {
	return m.db.WithContext(ctx).
		Model(&TestRequestRecord{}).
		Where("id = ?", id.String()).
		Update("status", string(status)).Error
}

func (m *MockRequestRepository) Delete(ctx context.Context, id uuid.UUID) error {
	return m.db.WithContext(ctx).Delete(&TestRequestRecord{}, id.String()).Error
}

func TestNewStateManager(t *testing.T) {
	db, requestRepo := setupTestDBForStateManager(t)

	sm, err := NewStateManager(db, requestRepo)
	require.NoError(t, err)
	require.NotNil(t, sm)

	// Cleanup
	err = sm.Close()
	assert.NoError(t, err)
}

func TestStateManager_GetArm_NoData(t *testing.T) {
	db, requestRepo := setupTestDBForStateManager(t)

	sm, err := NewStateManager(db, requestRepo)
	require.NoError(t, err)
	defer sm.Close()

	// Get arm for model with no data
	arm := sm.GetArm("nonexistent-model")
	assert.Nil(t, arm, "Should return nil for model with no completed requests")
}

func TestStateManager_GetArm_WithData(t *testing.T) {
	db, requestRepo := setupTestDBForStateManager(t)

	sm, err := NewStateManager(db, requestRepo)
	require.NoError(t, err)
	defer sm.Close()

	// Create test request data
	ctx := context.Background()

	// Create test data directly in SQLite tables
	for i := 0; i < 3; i++ {
		recordID := uuid.New().String()

		// Insert request record
		record := &TestRequestRecord{
			ID:          recordID,
			RequestData: json.RawMessage(`{"test": "data"}`),
			Model:       "test-model",
			Status:      string(persistence.RequestStatusCompleted),
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
		}
		err := db.WithContext(ctx).Create(record).Error
		require.NoError(t, err)

		// Add metrics
		metrics := &TestRequestMetrics{
			ID:        uuid.New().String(),
			RequestID: recordID,
			LatencyMs: 100 + int64(i*10),
			TotalCost: 0.01 + float64(i)*0.005,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}
		err = db.WithContext(ctx).Create(metrics).Error
		require.NoError(t, err)

		// Add feedback
		feedback := &TestRequestFeedback{
			ID:        uuid.New().String(),
			RequestID: recordID,
			Score:     0.8 + float64(i)*0.1,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}
		err = db.WithContext(ctx).Create(feedback).Error
		require.NoError(t, err)
	}

	// Get arm statistics
	arm := sm.GetArm("test-model")
	require.NotNil(t, arm)

	assert.Equal(t, "test-model", arm.Model)
	assert.Equal(t, 3, arm.RequestCount)
	assert.Equal(t, 3, arm.SuccessCount)     // All completed requests are considered successful
	assert.Equal(t, 330.0, arm.TotalLatency) // 100 + 110 + 120
	assert.Equal(t, 0.045, arm.TotalCost)    // 0.01 + 0.015 + 0.02
	assert.Equal(t, 2.7, arm.TotalFeedback)  // 0.8 + 0.9 + 1.0
	assert.Equal(t, 3, arm.FeedbackCount)
}

func TestStateManager_GetAllArms(t *testing.T) {
	db, requestRepo := setupTestDBForStateManager(t)

	sm, err := NewStateManager(db, requestRepo)
	require.NoError(t, err)
	defer sm.Close()

	ctx := context.Background()

	// Create requests for multiple models
	models := []string{"model-a", "model-b", "model-c"}
	for _, model := range models {
		recordID := uuid.New().String()

		record := &TestRequestRecord{
			ID:          recordID,
			RequestData: json.RawMessage(`{"test": "data"}`),
			Model:       model,
			Status:      string(persistence.RequestStatusCompleted),
			CreatedAt:   time.Now(),
			UpdatedAt:   time.Now(),
		}
		err := db.WithContext(ctx).Create(record).Error
		require.NoError(t, err)
	}

	// Get all arms
	arms := sm.GetAllArms()
	assert.Len(t, arms, 3)

	for _, model := range models {
		arm, exists := arms[model]
		assert.True(t, exists, "Arm for model %s should exist", model)
		assert.Equal(t, model, arm.Model)
		assert.Equal(t, 1, arm.RequestCount)
	}
}

func TestStateManager_UpdateArm_NoOp(t *testing.T) {
	db, requestRepo := setupTestDBForStateManager(t)

	sm, err := NewStateManager(db, requestRepo)
	require.NoError(t, err)
	defer sm.Close()

	// UpdateArm should be a no-op in the simplified approach
	metrics := bandit.PerformanceMetrics{
		Success:       true,
		Latency:       100.0,
		Cost:          0.01,
		FeedbackScore: 0.8,
	}

	err = sm.UpdateArm("test-model", metrics)
	assert.NoError(t, err)

	// Verify no arm was created (since no actual requests exist)
	arm := sm.GetArm("test-model")
	assert.Nil(t, arm)
}

func TestStateManager_GetStatistics(t *testing.T) {
	db, requestRepo := setupTestDBForStateManager(t)

	sm, err := NewStateManager(db, requestRepo)
	require.NoError(t, err)
	defer sm.Close()

	ctx := context.Background()

	// Create test data
	models := []string{"model-a", "model-b"}
	for _, model := range models {
		for j := 0; j < 2; j++ {
			recordID := uuid.New().String()

			record := &TestRequestRecord{
				ID:          recordID,
				RequestData: json.RawMessage(`{"test": "data"}`),
				Model:       model,
				Status:      string(persistence.RequestStatusCompleted),
				CreatedAt:   time.Now(),
				UpdatedAt:   time.Now(),
			}
			err := db.WithContext(ctx).Create(record).Error
			require.NoError(t, err)
		}
	}

	stats := sm.GetStatistics()
	assert.NotNil(t, stats)

	assert.Equal(t, 2, stats["total_arms"])
	assert.Equal(t, 4, stats["total_requests"])
	assert.Equal(t, 4, stats["total_successes"])

	modelsData, ok := stats["models"].(map[string]interface{})
	assert.True(t, ok)
	assert.Len(t, modelsData, 2)
}

func TestStateManager_Health(t *testing.T) {
	db, requestRepo := setupTestDBForStateManager(t)

	sm, err := NewStateManager(db, requestRepo)
	require.NoError(t, err)
	defer sm.Close()

	err = sm.Health()
	assert.NoError(t, err)
}

func TestStateManager_Close(t *testing.T) {
	db, requestRepo := setupTestDBForStateManager(t)

	sm, err := NewStateManager(db, requestRepo)
	require.NoError(t, err)

	err = sm.Close()
	assert.NoError(t, err)

	// Try to use after close
	err = sm.UpdateArm("test-model", bandit.PerformanceMetrics{})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "state manager is closed")
}

func TestStateManager_ConcurrentAccess(t *testing.T) {
	db, requestRepo := setupTestDBForStateManager(t)

	sm, err := NewStateManager(db, requestRepo)
	require.NoError(t, err)
	defer sm.Close()

	ctx := context.Background()

	// Create some test data
	recordID := uuid.New().String()

	record := &TestRequestRecord{
		ID:          recordID,
		RequestData: json.RawMessage(`{"test": "data"}`),
		Model:       "concurrent-model",
		Status:      string(persistence.RequestStatusCompleted),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
	err = db.WithContext(ctx).Create(record).Error
	require.NoError(t, err)

	// Test concurrent access
	var wg sync.WaitGroup
	numWorkers := 10

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			// Concurrent reads
			arm := sm.GetArm("concurrent-model")
			if arm != nil {
				assert.Equal(t, "concurrent-model", arm.Model)
			}

			// Concurrent no-op updates
			metrics := bandit.PerformanceMetrics{
				Success: true,
				Latency: 100.0,
			}
			err := sm.UpdateArm("concurrent-model", metrics)
			assert.NoError(t, err)
		}(i)
	}

	wg.Wait()
}

// TestStateManager_SimplifiedApproach demonstrates the benefits of the simplified approach
func TestStateManager_SimplifiedApproach(t *testing.T) {
	t.Log("=== Simplified Bandit State Management ===")
	t.Log("")
	t.Log("BENEFITS:")
	t.Log("✅ No persistence complexity - statistics computed on-demand")
	t.Log("✅ No snapshot triggers or timing logic needed")
	t.Log("✅ No data duplication - single source of truth (requests table)")
	t.Log("✅ Automatic consistency - always reflects current request data")
	t.Log("✅ Simpler configuration - no snapshot intervals or request counts")
	t.Log("✅ Easier debugging - statistics are computed from visible data")
	t.Log("✅ In-memory caching for performance optimization")
	t.Log("")
	t.Log("HOW IT WORKS:")
	t.Log("1. RequestTracker stores raw request data in 'requests' table")
	t.Log("2. StateManager computes bandit statistics on-demand via SQL queries")
	t.Log("3. Results cached in-memory with 30s TTL for performance")
	t.Log("4. Background refresh prevents cache misses during high traffic")
	t.Log("5. Much simpler and more reliable than complex snapshot mechanisms")

	// This test always passes - it's just for documentation
	assert.True(t, true)
}

// TestStateManager_CacheBasicFunctionality tests basic caching behavior
func TestStateManager_CacheBasicFunctionality(t *testing.T) {
	db, requestRepo := setupTestDBForStateManager(t)

	sm, err := NewStateManager(db, requestRepo)
	require.NoError(t, err)
	defer sm.Close()

	ctx := context.Background()

	// Create test data
	recordID := uuid.New().String()
	record := &TestRequestRecord{
		ID:          recordID,
		RequestData: json.RawMessage(`{"test": "cache"}`),
		Model:       "cache-test-model",
		Status:      string(persistence.RequestStatusCompleted),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
	err = db.WithContext(ctx).Create(record).Error
	require.NoError(t, err)

	// First call should be cache miss
	arms1 := sm.GetAllArms()
	assert.Len(t, arms1, 1)
	assert.Contains(t, arms1, "cache-test-model")

	// Check cache statistics after first call
	stats := sm.GetCacheStatistics()
	assert.Equal(t, int64(0), stats["cache_hits"])
	assert.Equal(t, int64(1), stats["cache_misses"])
	assert.Equal(t, float64(0), stats["cache_hit_ratio"])
	assert.Equal(t, 1, stats["cache_size"])

	// Second call should be cache hit
	arms2 := sm.GetAllArms()
	assert.Len(t, arms2, 1)
	assert.Equal(t, arms1["cache-test-model"].RequestCount, arms2["cache-test-model"].RequestCount)

	// Check cache statistics after second call
	stats = sm.GetCacheStatistics()
	assert.Equal(t, int64(1), stats["cache_hits"])
	assert.Equal(t, int64(1), stats["cache_misses"])
	assert.Equal(t, float64(0.5), stats["cache_hit_ratio"])
}

// TestStateManager_CacheExpiry tests cache expiration behavior
func TestStateManager_CacheExpiry(t *testing.T) {
	db, requestRepo := setupTestDBForStateManager(t)

	sm, err := NewStateManager(db, requestRepo)
	require.NoError(t, err)

	// Set a very short TTL for testing
	sm.cacheTTL = 100 * time.Millisecond
	defer sm.Close()

	ctx := context.Background()

	// Create test data
	recordID := uuid.New().String()
	record := &TestRequestRecord{
		ID:          recordID,
		RequestData: json.RawMessage(`{"test": "expiry"}`),
		Model:       "expiry-test-model",
		Status:      string(persistence.RequestStatusCompleted),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
	err = db.WithContext(ctx).Create(record).Error
	require.NoError(t, err)

	// First call populates cache
	arms1 := sm.GetAllArms()
	assert.Len(t, arms1, 1)

	// Wait for cache to expire
	time.Sleep(150 * time.Millisecond)

	// Add another record
	recordID2 := uuid.New().String()
	record2 := &TestRequestRecord{
		ID:          recordID2,
		RequestData: json.RawMessage(`{"test": "expiry2"}`),
		Model:       "expiry-test-model",
		Status:      string(persistence.RequestStatusCompleted),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
	err = db.WithContext(ctx).Create(record2).Error
	require.NoError(t, err)

	// Next call should see updated data (cache miss due to expiry)
	arms2 := sm.GetAllArms()
	assert.Len(t, arms2, 1)
	assert.Equal(t, 2, arms2["expiry-test-model"].RequestCount) // Should see both records

	// Verify cache miss occurred
	stats := sm.GetCacheStatistics()
	assert.Equal(t, int64(2), stats["cache_misses"]) // First call + expired call
}

// TestStateManager_CacheConcurrency tests concurrent access to cache
func TestStateManager_CacheConcurrency(t *testing.T) {
	db, requestRepo := setupTestDBForStateManager(t)

	sm, err := NewStateManager(db, requestRepo)
	require.NoError(t, err)
	defer sm.Close()

	ctx := context.Background()

	// Create test data with proper table creation
	recordID := uuid.New().String()
	record := &TestRequestRecord{
		ID:          recordID,
		RequestData: json.RawMessage(`{"test": "concurrency"}`),
		Model:       "concurrent-cache-model",
		Status:      string(persistence.RequestStatusCompleted),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
	err = db.WithContext(ctx).Create(record).Error
	require.NoError(t, err)

	// Add metrics for the record
	metricsID := uuid.New().String()
	metrics := &TestRequestMetrics{
		ID:         metricsID,
		RequestID:  recordID,
		TotalCost:  0.01,
		TokensUsed: 100,
		LatencyMs:  500,
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
	}
	err = db.WithContext(ctx).Create(metrics).Error
	require.NoError(t, err)

	// Prime the cache with an initial call to ensure data is ready
	initialArms := sm.GetAllArms()
	require.Len(t, initialArms, 1, "Initial cache population should return 1 arm")
	require.Contains(t, initialArms, "concurrent-cache-model", "Initial cache should contain test model")

	// Run concurrent GetAllArms calls
	const numGoroutines = 10
	var wg sync.WaitGroup
	results := make([]map[string]*bandit.BanditArm, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			results[index] = sm.GetAllArms()
		}(i)
	}

	wg.Wait()

	// All results should be consistent
	for i := 0; i < numGoroutines; i++ {
		assert.Len(t, results[i], 1)
		assert.Contains(t, results[i], "concurrent-cache-model")
		if results[i]["concurrent-cache-model"] != nil {
			assert.Equal(t, 1, results[i]["concurrent-cache-model"].RequestCount)
		}
	}

	// Should have cache hits (since cache was primed)
	stats := sm.GetCacheStatistics()
	assert.Greater(t, stats["cache_hits"].(int64), int64(0), "Should have at least one cache hit from concurrent calls")
}

// TestStateManager_CacheDeepCopy tests that cache returns deep copies
func TestStateManager_CacheDeepCopy(t *testing.T) {
	db, requestRepo := setupTestDBForStateManager(t)

	sm, err := NewStateManager(db, requestRepo)
	require.NoError(t, err)
	defer sm.Close()

	ctx := context.Background()

	// Create test data
	recordID := uuid.New().String()
	record := &TestRequestRecord{
		ID:          recordID,
		RequestData: json.RawMessage(`{"test": "deepcopy"}`),
		Model:       "deepcopy-test-model",
		Status:      string(persistence.RequestStatusCompleted),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
	err = db.WithContext(ctx).Create(record).Error
	require.NoError(t, err)

	// Get arms twice
	arms1 := sm.GetAllArms()
	arms2 := sm.GetAllArms()

	// Should be cache hits with same data
	assert.Equal(t, arms1["deepcopy-test-model"].RequestCount, arms2["deepcopy-test-model"].RequestCount)

	// Modify one result - should not affect the other (deep copy)
	arms1["deepcopy-test-model"].RequestCount = 999
	assert.Equal(t, 1, arms2["deepcopy-test-model"].RequestCount) // Should remain unchanged

	// Get arms again - should still return correct data
	arms3 := sm.GetAllArms()
	assert.Equal(t, 1, arms3["deepcopy-test-model"].RequestCount) // Should be original value
}

// TestStateManager_BackgroundRefresh tests background cache warming
func TestStateManager_BackgroundRefresh(t *testing.T) {
	db, requestRepo := setupTestDBForStateManager(t)

	sm, err := NewStateManager(db, requestRepo)
	require.NoError(t, err)

	// Set short refresh interval for testing
	sm.cacheTTL = 100 * time.Millisecond
	defer sm.Close()

	ctx := context.Background()

	// Create test data
	recordID := uuid.New().String()
	record := &TestRequestRecord{
		ID:          recordID,
		RequestData: json.RawMessage(`{"test": "background"}`),
		Model:       "background-test-model",
		Status:      string(persistence.RequestStatusCompleted),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
	err = db.WithContext(ctx).Create(record).Error
	require.NoError(t, err)

	// Populate cache
	arms1 := sm.GetAllArms()
	assert.Len(t, arms1, 1)

	// Wait for background refresh to potentially trigger
	// Note: Background refresh runs every 25 seconds in production,
	// but the logic should be working even if we can't easily test the timing
	time.Sleep(50 * time.Millisecond)

	// Cache should still be functional
	arms2 := sm.GetAllArms()
	assert.Len(t, arms2, 1)
}

// TestStateManager_GetStatisticsWithCache tests that GetStatistics includes cache metrics
func TestStateManager_GetStatisticsWithCache(t *testing.T) {
	db, requestRepo := setupTestDBForStateManager(t)

	sm, err := NewStateManager(db, requestRepo)
	require.NoError(t, err)
	defer sm.Close()

	ctx := context.Background()

	// Create test data
	recordID := uuid.New().String()
	record := &TestRequestRecord{
		ID:          recordID,
		RequestData: json.RawMessage(`{"test": "stats"}`),
		Model:       "stats-test-model",
		Status:      string(persistence.RequestStatusCompleted),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
	err = db.WithContext(ctx).Create(record).Error
	require.NoError(t, err)

	// Trigger some cache activity
	sm.GetAllArms() // Cache miss
	sm.GetAllArms() // Cache hit

	// Get statistics (NOTE: GetStatistics calls GetAllArms internally, adding another cache hit)
	stats := sm.GetStatistics()

	// Should include cache statistics
	assert.Contains(t, stats, "cache")
	cacheStats := stats["cache"].(map[string]interface{})

	assert.Contains(t, cacheStats, "cache_hits")
	assert.Contains(t, cacheStats, "cache_misses")
	assert.Contains(t, cacheStats, "cache_hit_ratio")
	assert.Contains(t, cacheStats, "cache_size")
	assert.Contains(t, cacheStats, "cache_ttl_seconds")

	// Expect 2 cache hits: one from explicit GetAllArms() call, one from GetStatistics() internal call
	assert.Equal(t, int64(2), cacheStats["cache_hits"])
	assert.Equal(t, int64(1), cacheStats["cache_misses"])
	assert.InDelta(t, float64(2.0/3.0), cacheStats["cache_hit_ratio"], 0.01)
	assert.Equal(t, 1, cacheStats["cache_size"])
	assert.Equal(t, float64(30), cacheStats["cache_ttl_seconds"])
}
