package httpiface

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"

	domain "llm-router/domain/chat"
	"llm-router/domain/persistence"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

type ChatService interface {
	Chat(ctx context.Context, req *domain.Request) (*domain.Response, error)
	Stream(ctx context.Context, req *domain.Request, onChunk domain.StreamHandler[domain.StreamChunk]) error
}

type ContextualRoutingFactory interface {
	Health(ctx context.Context) error
	Readiness(ctx context.Context) error
}

type Router struct {
	service           ChatService
	corsOrigins       []string
	tracker           persistence.RequestTracker
	metricsRepo       persistence.MetricsRepository
	requestRepo       persistence.RequestRepository
	dbManager         persistence.DatabaseManager
	processor         persistence.EventProcessor
	contextualFactory ContextualRoutingFactory
}

func NewRouter(service ChatService, corsOrigins []string) *Router {
	return &Router{
		service:     service,
		corsOrigins: corsOrigins,
	}
}

// NewRouterWithPersistence creates a router with persistence capabilities
func NewRouterWithPersistence(
	service ChatService,
	corsOrigins []string,
	tracker persistence.RequestTracker,
	metricsRepo persistence.MetricsRepository,
	requestRepo persistence.RequestRepository,
	dbManager persistence.DatabaseManager,
	processor persistence.EventProcessor,
	contextualFactory ContextualRoutingFactory,
) *Router {
	return &Router{
		service:           service,
		corsOrigins:       corsOrigins,
		tracker:           tracker,
		metricsRepo:       metricsRepo,
		requestRepo:       requestRepo,
		dbManager:         dbManager,
		processor:         processor,
		contextualFactory: contextualFactory,
	}
}

func (r *Router) SetupRoutes() *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Logger())
	router.Use(gin.Recovery())
	router.Use(r.corsMiddleware())

	// Health endpoints - no request ID required for monitoring tools
	router.GET("/live", r.liveness)
	router.GET("/ready", r.readiness)
	router.GET("/health", r.healthCheck)

	// Business API endpoints - require request ID for tracking
	api := router.Group("/")
	api.Use(r.requestIDMiddleware())
	api.POST("/chat/completions", r.chatCompletions)

	// Persistence endpoints (only available if repositories are configured)
	if r.tracker != nil && r.metricsRepo != nil && r.requestRepo != nil {
		api.POST("/feedback", r.submitFeedback)
		api.GET("/metrics/:request-id", r.getRequestMetrics)
		api.GET("/metrics", r.getAggregatedMetrics)
		api.GET("/requests/:request-id", r.getRequest)
	}

	return router
}

func (r *Router) corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		reqOrigin := c.GetHeader("Origin")
		if reqOrigin == "" {
			// Maintain previous behavior used by tests: join configured origins
			c.Header("Access-Control-Allow-Origin", strings.Join(r.corsOrigins, ", "))
		} else {
			allowOrigin := ""
			if len(r.corsOrigins) == 1 && r.corsOrigins[0] == "*" {
				allowOrigin = "*"
			} else {
				for _, allowed := range r.corsOrigins {
					if allowed == reqOrigin {
						allowOrigin = reqOrigin
						break
					}
				}
			}
			if allowOrigin != "" {
				c.Header("Access-Control-Allow-Origin", allowOrigin)
			}
		}
		c.Header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}
		c.Next()
	}
}

func (r *Router) requestIDMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Check for existing request ID from client
		clientRequestID := c.GetHeader("X-Request-ID")
		clientCorrelationID := c.GetHeader("X-Correlation-ID")

		var requestUUID uuid.UUID
		var requestID string

		// Require client-provided ID for proper tracking
		if clientRequestID != "" {
			// Try to parse as UUID first
			if parsedUUID, err := uuid.Parse(clientRequestID); err == nil {
				requestUUID = parsedUUID
				requestID = clientRequestID
			} else {
				// If not a UUID, generate a new one but keep the client ID for reference
				requestUUID = uuid.New()
				requestID = requestUUID.String()
				c.Header("X-Client-Request-ID", clientRequestID) // Echo back original
			}
		} else if clientCorrelationID != "" {
			// Use correlation ID if available
			if parsedUUID, err := uuid.Parse(clientCorrelationID); err == nil {
				requestUUID = parsedUUID
				requestID = clientCorrelationID
			} else {
				requestUUID = uuid.New()
				requestID = requestUUID.String()
				c.Header("X-Client-Correlation-ID", clientCorrelationID)
			}
		} else {
			// Reject requests without request ID for proper tracking
			c.JSON(http.StatusBadRequest, gin.H{
				"error":   "Missing required header: X-Request-ID or X-Correlation-ID",
				"message": "All requests must include a request ID for proper tracking and feedback persistence",
			})
			c.Abort()
			return
		}

		// Set response headers
		c.Header("X-Request-ID", requestID)
		c.Header("X-Request-UUID", requestUUID.String())

		// Store in context for internal use
		c.Set("request_uuid", requestUUID.String())
		c.Set("request_id_string", requestID)

		c.Next()
	}
}

func (r *Router) healthCheck(c *gin.Context) {
	checks := gin.H{
		"api": "ok",
	}

	overallOK := true

	if r.dbManager != nil {
		if err := r.dbManager.Health(c.Request.Context()); err != nil {
			checks["db"] = gin.H{"ok": false, "error": err.Error()}
			overallOK = false
		} else {
			checks["db"] = gin.H{"ok": true}
		}
	}

	if r.processor != nil {
		ph := r.processor.Health()
		checks["processor"] = ph
		// consider queue size too large as degraded
		if !ph.IsRunning {
			overallOK = false
		}
	}

	// Embedding service health check if configured
	if r.contextualFactory != nil {
		if err := r.contextualFactory.Health(c.Request.Context()); err != nil {
			checks["embedding_service"] = gin.H{"ok": false, "error": err.Error()}
			overallOK = false
		} else {
			checks["embedding_service"] = gin.H{"ok": true}
		}
	}

	status := "healthy"
	code := http.StatusOK
	if !overallOK {
		status = "degraded"
		code = http.StatusServiceUnavailable
	}

	health := gin.H{
		"status":    status,
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"service":   "llm-proxy-router",
		"version":   "1.0.0",
		"checks":    checks,
	}
	c.JSON(code, health)
}

// liveness probe: process is up and serving HTTP
func (r *Router) liveness(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":    "alive",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	})
}

// readiness probe: dependencies healthy and ready to serve traffic
func (r *Router) readiness(c *gin.Context) {
	checks := gin.H{}
	ready := true

	// DB readiness if configured
	if r.dbManager != nil {
		if err := r.dbManager.Health(c.Request.Context()); err != nil {
			checks["db"] = gin.H{"ok": false, "error": err.Error()}
			ready = false
		} else {
			checks["db"] = gin.H{"ok": true}
		}
	}

	// Processor readiness if configured
	if r.processor != nil {
		ph := r.processor.Health()
		checks["processor"] = ph
		if !ph.IsRunning {
			ready = false
		}
	}

	// Embedding service readiness if configured
	if r.contextualFactory != nil {
		if err := r.contextualFactory.Readiness(c.Request.Context()); err != nil {
			checks["embedding_service"] = gin.H{"ok": false, "error": err.Error()}
			ready = false
		} else {
			checks["embedding_service"] = gin.H{"ok": true}
		}
	}

	if ready {
		c.JSON(http.StatusOK, gin.H{
			"status":    "ready",
			"timestamp": time.Now().UTC().Format(time.RFC3339),
			"checks":    checks,
		})
		return
	}
	c.JSON(http.StatusServiceUnavailable, gin.H{
		"status":    "not_ready",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"checks":    checks,
	})
}

func (r *Router) chatCompletions(c *gin.Context) {
	var req domain.Request
	if err := c.ShouldBindJSON(&req); err != nil {
		logrus.WithError(err).Error("Failed to bind request")
		c.JSON(http.StatusBadRequest, domain.ErrorResponse{Error: "Invalid request format"})
		return
	}

	if len(req.Messages) == 0 {
		c.JSON(http.StatusBadRequest, domain.ErrorResponse{Error: "Messages cannot be empty"})
		return
	}

	if req.Stream {
		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		c.Header("Connection", "keep-alive")
		c.Status(http.StatusOK)
		flusher, ok := c.Writer.(http.Flusher)
		if !ok {
			c.JSON(http.StatusInternalServerError, domain.ErrorResponse{Error: "Streaming not supported by server"})
			return
		}
		// Capture usage from the final chunk (OpenRouter sends usage on last chunk)
		var finalUsage *domain.Usage
		// Create context with Gin values for the streaming service
		streamCtx := c.Request.Context()
		if requestUUID, exists := c.Get("request_uuid"); exists {
			streamCtx = context.WithValue(streamCtx, "request_uuid", requestUUID)
		}
		if requestIDString, exists := c.Get("request_id_string"); exists {
			streamCtx = context.WithValue(streamCtx, "request_id_string", requestIDString)
		}

		if err := r.service.Stream(streamCtx, &req, func(chunk domain.StreamChunk) error {
			data, err := json.Marshal(chunk)
			if err != nil {
				return err
			}
			if chunk.Usage != nil {
				finalUsage = chunk.Usage
			}
			if _, err := c.Writer.Write([]byte("data: ")); err != nil {
				return err
			}
			if _, err := c.Writer.Write(data); err != nil {
				return err
			}
			if _, err := c.Writer.Write([]byte("\n\n")); err != nil {
				return err
			}
			flusher.Flush()
			return nil
		}); err != nil {
			logrus.WithError(err).Error("Streaming failed")
			return
		}
		c.Writer.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
		// Instrument usage if available
		requestIDVal, _ := c.Get("request_id")
		if finalUsage != nil {
			logrus.WithFields(logrus.Fields{
				"request_id":       requestIDVal,
				"usage_total":      finalUsage.TotalTokens,
				"usage_prompt":     finalUsage.PromptTokens,
				"usage_completion": finalUsage.CompletionTokens,
				"streaming":        true,
			}).Info("Chat usage")
		} else {
			logrus.WithFields(logrus.Fields{
				"request_id": requestIDVal,
				"streaming":  true,
			}).Warn("No usage reported on stream end")
		}
		return
	}

	// Create context with Gin values for the chat service
	ctx := c.Request.Context()
	if requestUUID, exists := c.Get("request_uuid"); exists {
		ctx = context.WithValue(ctx, "request_uuid", requestUUID)
	}
	if requestIDString, exists := c.Get("request_id_string"); exists {
		ctx = context.WithValue(ctx, "request_id_string", requestIDString)
	}

	resp, err := r.service.Chat(ctx, &req)
	if err != nil {
		logrus.WithError(err).Error("Failed to process chat completion")
		c.JSON(http.StatusInternalServerError, domain.ErrorResponse{Error: "Failed to process request"})
		return
	}
	// Instrument usage for non-streaming
	requestIDVal, _ := c.Get("request_id")
	logrus.WithFields(logrus.Fields{
		"request_id":       requestIDVal,
		"usage_total":      resp.Usage.TotalTokens,
		"usage_prompt":     resp.Usage.PromptTokens,
		"usage_completion": resp.Usage.CompletionTokens,
		"streaming":        false,
	}).Info("Chat usage")

	c.JSON(http.StatusOK, resp)
}

// FeedbackRequest represents the structure for feedback submission
type FeedbackRequest struct {
	RequestID    string  `json:"request_id" binding:"required"`
	FeedbackText string  `json:"feedback_text"`
	Score        float64 `json:"score" binding:"min=0,max=1"`
}

// submitFeedback handles feedback submission for requests
func (r *Router) submitFeedback(c *gin.Context) {
	if r.tracker == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Feedback system not available"})
		return
	}

	var req FeedbackRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format", "details": err.Error()})
		return
	}

	requestID, err := uuid.Parse(req.RequestID)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request ID format"})
		return
	}

	// Submit feedback asynchronously with timeout
	go func(parentCtx context.Context) {
		opCtx, cancel := context.WithTimeout(context.WithoutCancel(parentCtx), 5*time.Second)
		if err := r.tracker.SubmitFeedback(opCtx, requestID, req.FeedbackText, req.Score); err != nil {
			logrus.WithError(err).Errorf("Failed to submit feedback for request %s", requestID)
		}
		cancel()
	}(c.Request.Context())

	c.JSON(http.StatusAccepted, gin.H{
		"message":    "Feedback submitted successfully",
		"request_id": req.RequestID,
	})
}

// getRequestMetrics retrieves metrics for a specific request
func (r *Router) getRequestMetrics(c *gin.Context) {
	if r.metricsRepo == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Metrics system not available"})
		return
	}

	requestIDStr := c.Param("request-id")
	requestID, err := uuid.Parse(requestIDStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request ID format"})
		return
	}

	metrics, err := r.metricsRepo.FindByRequestID(c.Request.Context(), requestID)
	if err != nil {
		logrus.WithError(err).Errorf("Failed to get metrics for request %s", requestID)
		c.JSON(http.StatusNotFound, gin.H{"error": "Metrics not found for the specified request"})
		return
	}

	c.JSON(http.StatusOK, metrics)
}

// getAggregatedMetrics retrieves aggregated metrics across all requests
func (r *Router) getAggregatedMetrics(c *gin.Context) {
	if r.metricsRepo == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Metrics system not available"})
		return
	}

	// Parse optional limit parameter
	limitStr := c.DefaultQuery("limit", "1000")
	limit, err := strconv.Atoi(limitStr)
	if err != nil || limit < 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid limit parameter"})
		return
	}

	metrics, err := r.metricsRepo.GetAggregatedMetrics(c.Request.Context(), limit)
	if err != nil {
		logrus.WithError(err).Error("Failed to get aggregated metrics")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve aggregated metrics"})
		return
	}

	c.JSON(http.StatusOK, metrics)
}

// getRequest retrieves a complete request record with all relations
func (r *Router) getRequest(c *gin.Context) {
	if r.requestRepo == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Request storage system not available"})
		return
	}

	requestIDStr := c.Param("request-id")
	requestID, err := uuid.Parse(requestIDStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request ID format"})
		return
	}

	record, err := r.requestRepo.FindByIDWithRelations(c.Request.Context(), requestID)
	if err != nil {
		logrus.WithError(err).Errorf("Failed to get request %s", requestID)
		c.JSON(http.StatusNotFound, gin.H{"error": "Request not found"})
		return
	}

	c.JSON(http.StatusOK, record)
}
