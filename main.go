package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	appchat "llm-router/application/chat"
	"llm-router/infrastructure/openrouter"
	infrapersistence "llm-router/infrastructure/persistence"
	"llm-router/infrastructure/routing"
	httpiface "llm-router/interfaces/http"
	"llm-router/internal/config"

	"github.com/sirupsen/logrus"
)

func main() {
	ctx := context.Background()

	cfg, err := config.LoadYAML("")
	if err != nil {
		logrus.WithError(err).Fatal("Failed to load configuration")
	}

	// Configure logging level
	level, err := logrus.ParseLevel(cfg.Logging.Level)
	if err != nil {
		level = logrus.InfoLevel
	}
	logrus.SetLevel(level)

	// Configure logging formatter per environment
	switch cfg.Logging.Format {
	case "json":
		logrus.SetFormatter(&logrus.JSONFormatter{TimestampFormat: time.RFC3339Nano})
	case "text":
		logrus.SetFormatter(&logrus.TextFormatter{FullTimestamp: true})
	case "auto", "":
		// Default to text for development-friendly output
		logrus.SetFormatter(&logrus.TextFormatter{FullTimestamp: true})
	default:
		logrus.SetFormatter(&logrus.TextFormatter{FullTimestamp: true})
	}

	// Optionally include caller info
	logrus.SetReportCaller(cfg.Logging.ReportCaller)

	logrus.WithFields(logrus.Fields{
		"port":               cfg.Server.Port,
		"host":               cfg.Server.Host,
		"models":             cfg.LLMProvider.AllowedModels,
		"enable_persistence": cfg.Database.EnablePersistence,
		"embedding_service":  cfg.Embedding.ServiceType,
	}).Info("Starting LLM-Proxy Router")

	// Create base provider
	baseProvider := openrouter.NewProvider(cfg.LLMProvider.APIKey, cfg.LLMProvider.BaseURL, cfg.LLMProvider.AllowedModels, cfg.Server.RefererURL, cfg.Server.AppName)

	// Wrap with circuit breaker for resilience
	circuitBreakerConfig := openrouter.CircuitBreakerConfig{
		Enabled:          cfg.CircuitBreaker.Enabled,
		FailureThreshold: cfg.CircuitBreaker.FailureThreshold,
		SuccessThreshold: cfg.CircuitBreaker.SuccessThreshold,
		Timeout:          cfg.CircuitBreaker.Timeout,
		MaxRequests:      cfg.CircuitBreaker.MaxRequests,
	}
	provider := openrouter.NewCircuitBreakerProvider(baseProvider, baseProvider, circuitBreakerConfig)

	logrus.WithFields(logrus.Fields{
		"enabled":           circuitBreakerConfig.Enabled,
		"failure_threshold": circuitBreakerConfig.FailureThreshold,
		"timeout":           circuitBreakerConfig.Timeout,
	}).Info("Circuit breaker configured")

	var service *appchat.Service
	var router *httpiface.Router
	var dbManager *infrapersistence.DatabaseManager
	var eventProcessor *infrapersistence.EventProcessor
	var contextualFactory *routing.ContextualRoutingFactory

	if cfg.Database.EnablePersistence {
		// Initialize database components
		dbManager = infrapersistence.NewDatabaseManager()

		// Connect to database
		if err := dbManager.Connect(ctx, cfg.GetDatabaseDSN()); err != nil {
			logrus.WithError(err).Fatal("Failed to connect to database")
		}

		// Run migrations
		if err := dbManager.Migrate(); err != nil {
			logrus.WithError(err).Fatal("Failed to run database migrations")
		}

		// Get repositories
		requestRepo, metricsRepo, feedbackRepo := dbManager.GetRepositories()

		// Initialize event processor
		eventProcessor = infrapersistence.NewEventProcessor(
			requestRepo,
			metricsRepo,
			feedbackRepo,
			cfg.Database.Workers,
			cfg.Database.BufferSize,
		)

		// Start event processor
		if err := eventProcessor.Start(ctx); err != nil {
			logrus.WithError(err).Fatal("Failed to start event processor")
		}

		// Create request tracker
		tracker := infrapersistence.NewRequestTracker(eventProcessor)

		// Initialize contextual routing
		routingConfig := routing.ContextualRoutingConfig{
			EmbeddingServiceType: cfg.Embedding.ServiceType,
			EmbeddingServiceURL:  cfg.Embedding.ServiceURL,
			EmbeddingModelPath:   cfg.Embedding.ModelPath,
			MaxEmbeddingWorkers:  cfg.Embedding.MaxWorkers,
			EmbeddingCacheSize:   cfg.Embedding.CacheSize,
			EmbeddingTimeout:     cfg.Embedding.TimeoutMs,
			SimilarityThreshold:  cfg.Bandit.Similarity.Threshold,
			MaxSimilarRequests:   cfg.Bandit.Similarity.MaxSimilarRequests,
			RecencyDays:          cfg.Bandit.Similarity.RecencyDays,
			FeedbackWeight:       cfg.Bandit.ThompsonSampling.FeedbackWeight,
			LatencyWeight:        cfg.Bandit.ThompsonSampling.LatencyWeight,
			CostWeight:           cfg.Bandit.ThompsonSampling.CostWeight,
			ExplorationRate:      cfg.Bandit.ThompsonSampling.ExplorationRate,
			MinSimilarRequests:   cfg.Bandit.Similarity.MinSimilarRequests,
			DefaultModel:         cfg.Bandit.DefaultModel,
			BatchSize:            cfg.Bandit.Persistence.BatchSize,

			// Cold start configuration
			MinConfidenceScore:   cfg.Bandit.ColdStart.MinConfidenceScore,
			OptimisticPrior:      cfg.Bandit.ColdStart.OptimisticPrior,
			ExplorationBonus:     cfg.Bandit.ColdStart.ExplorationBonus,
			MinRequestsForGlobal: cfg.Bandit.ColdStart.MinRequestsForGlobal,
		}

		contextualFactory, err = routing.NewContextualRoutingFactory(routingConfig, dbManager.GetDB(), requestRepo)
		if err != nil {
			logrus.WithError(err).Fatal("Failed to initialize contextual routing")
		}

		// Attach contextual router to provider (works with circuit breaker wrapper)
		contextualFactory.AttachToProviderInterface(provider)

		// Get embedding service and bandit router from contextual factory
		embeddingService := contextualFactory.GetEmbeddingService()
		banditRouter := contextualFactory.GetRouter()

		// Create service with tracking, embedding service, and bandit router
		service = appchat.NewService(provider, provider, tracker, embeddingService, banditRouter)

		// Create router with persistence and health sources
		router = httpiface.NewRouterWithPersistence(service, cfg.Server.CorsOrigins, tracker, metricsRepo, requestRepo, dbManager, eventProcessor, contextualFactory)

		logrus.Info("Persistence layer initialized successfully")
	} else {
		// Create service without tracking
		service = appchat.NewServiceWithoutTracking(provider, provider)

		// Create router without persistence
		router = httpiface.NewRouter(service, cfg.Server.CorsOrigins)

		logrus.Info("Running without persistence layer")
	}

	ginRouter := router.SetupRoutes()

	address := fmt.Sprintf("%s:%s", cfg.Server.Host, cfg.Server.Port)
	server := &http.Server{
		Addr:              address,
		Handler:           ginRouter,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	// Channel to listen for interrupt signal to trigger shutdown
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)

	// Start server in a goroutine
	go func() {
		logrus.WithField("address", address).Info("Server starting")
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logrus.WithError(err).Fatal("Failed to start server")
		}
	}()

	// Block until signal is received
	<-c
	logrus.Info("Shutting down server...")

	// Create a deadline for shutdown
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	// Attempt graceful shutdown
	if err := server.Shutdown(shutdownCtx); err != nil {
		logrus.WithError(err).Error("Server forced to shutdown")
	} else {
		logrus.Info("Server shutdown complete")
	}

	// Clean up persistence layer if initialized
	if cfg.Database.EnablePersistence {
		logrus.Info("Shutting down persistence layer...")

		if contextualFactory != nil {
			if err := contextualFactory.Close(); err != nil {
				logrus.WithError(err).Error("Failed to close contextual routing factory")
			}
		}

		if eventProcessor != nil {
			if err := eventProcessor.Stop(); err != nil {
				logrus.WithError(err).Error("Failed to stop event processor")
			}
		}

		if dbManager != nil {
			if err := dbManager.Close(); err != nil {
				logrus.WithError(err).Error("Failed to close database connection")
			}
		}

		logrus.Info("Persistence layer shutdown complete")
	}
}
