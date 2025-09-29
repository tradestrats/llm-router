package persistence

import (
	"context"
	"fmt"
	"time"

	"llm-router/domain/persistence"

	"github.com/sirupsen/logrus"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

// DatabaseManager implements the persistence.DatabaseManager interface
type DatabaseManager struct {
	db           *gorm.DB
	requestRepo  persistence.RequestRepository
	metricsRepo  persistence.MetricsRepository
	feedbackRepo persistence.FeedbackRepository
}

// NewDatabaseManager creates a new database manager instance
func NewDatabaseManager() *DatabaseManager {
	return &DatabaseManager{}
}

// Connect establishes database connection
func (dm *DatabaseManager) Connect(ctx context.Context, dsn string) error {
	logrus.Info("Connecting to PostgreSQL database...")

	// Configure GORM logger
	gormLogger := logger.New(
		logrus.StandardLogger(),
		logger.Config{
			SlowThreshold:             200 * time.Millisecond,
			LogLevel:                  logger.Warn,
			IgnoreRecordNotFoundError: true,
			Colorful:                  false,
		},
	)

	// Connect to database
	db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{
		Logger: gormLogger,
		NowFunc: func() time.Time {
			return time.Now().UTC()
		},
	})
	if err != nil {
		return fmt.Errorf("failed to connect to database: %w", err)
	}

	// Get underlying SQL DB for connection pool configuration
	sqlDB, err := db.DB()
	if err != nil {
		return fmt.Errorf("failed to get underlying SQL DB: %w", err)
	}

	// Configure connection pool
	sqlDB.SetMaxIdleConns(10)
	sqlDB.SetMaxOpenConns(100)
	sqlDB.SetConnMaxLifetime(time.Hour)

	// Test connection
	if err := sqlDB.PingContext(ctx); err != nil {
		return fmt.Errorf("failed to ping database: %w", err)
	}

	dm.db = db

	// Initialize repositories
	dm.requestRepo = NewRequestRepository(db)
	dm.metricsRepo = NewMetricsRepository(db)
	dm.feedbackRepo = NewFeedbackRepository(db)

	logrus.Info("Successfully connected to PostgreSQL database")
	return nil
}

// Close closes the database connection
func (dm *DatabaseManager) Close() error {
	if dm.db == nil {
		return nil
	}

	sqlDB, err := dm.db.DB()
	if err != nil {
		return fmt.Errorf("failed to get underlying SQL DB for close: %w", err)
	}

	if err := sqlDB.Close(); err != nil {
		return fmt.Errorf("failed to close database connection: %w", err)
	}

	logrus.Info("Database connection closed successfully")
	return nil
}

// Migrate runs database migrations
func (dm *DatabaseManager) Migrate() error {
	if dm.db == nil {
		return fmt.Errorf("database connection not established")
	}

	logrus.Info("Running database migrations...")

	// Enable required extensions
	if err := dm.db.Exec("CREATE EXTENSION IF NOT EXISTS \"pgcrypto\"").Error; err != nil {
		return fmt.Errorf("failed to create pgcrypto extension: %w", err)
	}
	if err := dm.db.Exec("CREATE EXTENSION IF NOT EXISTS vector").Error; err != nil {
		return fmt.Errorf("failed to create pgvector extension: %w", err)
	}

	// Create tables manually to handle pgvector fields properly
	if err := dm.createTables(); err != nil {
		return fmt.Errorf("failed to create tables: %w", err)
	}

	// Create indexes for performance
	if err := dm.createIndexes(); err != nil {
		return fmt.Errorf("failed to create indexes: %w", err)
	}

	logrus.Info("Database migrations completed successfully")
	return nil
}

// createTables creates database tables manually
func (dm *DatabaseManager) createTables() error {
	// Create requests table
	if err := dm.db.Exec(`
		CREATE TABLE IF NOT EXISTS requests (
			id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
			request_data JSONB NOT NULL,
			response_data JSONB,
			model VARCHAR(255) NOT NULL,
			is_streaming BOOLEAN DEFAULT false,
			status VARCHAR(50) DEFAULT 'pending' NOT NULL,
			embedding VECTOR(384),
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
		)
	`).Error; err != nil {
		return fmt.Errorf("failed to create requests table: %w", err)
	}

	// Create request_metrics table
	if err := dm.db.Exec(`
		CREATE TABLE IF NOT EXISTS request_metrics (
			id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
			request_id UUID NOT NULL REFERENCES requests(id) ON DELETE CASCADE,
			total_cost DECIMAL(10,6) DEFAULT 0,
			tokens_used INTEGER DEFAULT 0,
			latency_ms BIGINT DEFAULT 0,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
		)
	`).Error; err != nil {
		return fmt.Errorf("failed to create request_metrics table: %w", err)
	}

	// Create request_feedback table
	if err := dm.db.Exec(`
		CREATE TABLE IF NOT EXISTS request_feedback (
			id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
			request_id UUID NOT NULL REFERENCES requests(id) ON DELETE CASCADE,
			feedback_text TEXT,
			score DECIMAL(3,2) CHECK (score >= 0 AND score <= 1),
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
		)
	`).Error; err != nil {
		return fmt.Errorf("failed to create request_feedback table: %w", err)
	}

	return nil
}

// createIndexes creates additional database indexes for performance
func (dm *DatabaseManager) createIndexes() error {
	indexes := []string{
		// Basic indexes (existing)
		"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_requests_model_created ON requests (model, created_at DESC)",
		"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_requests_status_created ON requests (status, created_at DESC)",
		"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_request_metrics_created ON request_metrics (created_at DESC)",
		"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_request_feedback_created ON request_feedback (created_at DESC)",
		"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_request_feedback_score ON request_feedback (score DESC)",
		// Ensure one metrics row per request for faster lookups and data integrity
		"CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS ux_request_metrics_request_id ON request_metrics (request_id)",
		// Speed up feedback lookup by request_id then recency
		"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_request_feedback_request_created ON request_feedback (request_id, created_at DESC)",
		// HNSW index for embedding similarity search (only create if embeddings exist)
		"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_requests_embedding_cosine ON requests USING hnsw (embedding vector_cosine_ops) WHERE embedding IS NOT NULL",
		// Composite index for embedding similarity with recency filter
		"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_requests_embedding_created ON requests (created_at DESC) WHERE embedding IS NOT NULL",

		// Bandit Performance Optimization Indexes
		// Covering index for requests - eliminates heap lookups for bandit queries
		"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_requests_bandit_covering ON requests (model, status) INCLUDE (id, created_at) WHERE status = 'completed'",
		// Covering index for metrics - avoids heap lookups during JOINs
		"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_metrics_bandit_covering ON request_metrics (request_id) INCLUDE (latency_ms, total_cost, tokens_used)",
		// Covering index for feedback with positive score filter
		"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feedback_bandit_covering ON request_feedback (request_id) INCLUDE (score, created_at) WHERE score > 0",
		// Partial index for completed requests only (reduces index size by ~90%)
		"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_requests_completed_model ON requests (model, id) WHERE status = 'completed'",
		// Optimized index for GetAllArms() GROUP BY operations
		"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_requests_all_models ON requests (status, model, id) WHERE status = 'completed'",
		// Foreign key optimization for metrics JOINs
		"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_metrics_request_composite ON request_metrics (request_id, latency_ms, total_cost)",
		// Foreign key optimization for feedback JOINs with positive scores
		"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_feedback_request_composite ON request_feedback (request_id, score) WHERE score > 0",
		// Recent requests optimization (30-day window for hot data)
		"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_requests_recent_model ON requests (model, created_at DESC, id) WHERE created_at > NOW() - INTERVAL '30 days' AND status = 'completed'",
	}

	for _, index := range indexes {
		if err := dm.db.Exec(index).Error; err != nil {
			logrus.WithError(err).Warnf("Failed to create index: %s", index)
			// Continue with other indexes even if one fails
		}
	}

	return nil
}

// Health checks database connectivity
func (dm *DatabaseManager) Health(ctx context.Context) error {
	if dm.db == nil {
		return fmt.Errorf("database connection not established")
	}

	sqlDB, err := dm.db.DB()
	if err != nil {
		return fmt.Errorf("failed to get underlying SQL DB: %w", err)
	}

	if err := sqlDB.PingContext(ctx); err != nil {
		return fmt.Errorf("database ping failed: %w", err)
	}

	return nil
}

// GetRepositories returns initialized repositories
func (dm *DatabaseManager) GetRepositories() (persistence.RequestRepository, persistence.MetricsRepository, persistence.FeedbackRepository) {
	return dm.requestRepo, dm.metricsRepo, dm.feedbackRepo
}

// GetDB returns the underlying GORM database instance
func (dm *DatabaseManager) GetDB() *gorm.DB {
	return dm.db
}

// WithTransaction executes a function within a database transaction
func (dm *DatabaseManager) WithTransaction(ctx context.Context, fn func(ctx context.Context) error) error {
	if dm.db == nil {
		return fmt.Errorf("database connection not established")
	}

	tx := dm.db.WithContext(ctx).Begin()
	if tx.Error != nil {
		return fmt.Errorf("failed to begin transaction: %w", tx.Error)
	}

	defer func() {
		if r := recover(); r != nil {
			tx.Rollback()
			panic(r)
		}
	}()

	// Create a new context that includes the transaction
	txCtx := context.WithValue(ctx, "gorm_tx", tx)

	if err := fn(txCtx); err != nil {
		if rbErr := tx.Rollback().Error; rbErr != nil {
			logrus.WithError(rbErr).Error("Failed to rollback transaction")
		}
		return err
	}

	if err := tx.Commit().Error; err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}
