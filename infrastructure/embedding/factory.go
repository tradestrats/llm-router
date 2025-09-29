package embedding

import (
	"fmt"

	"llm-router/domain/embedding"

	"github.com/sirupsen/logrus"
)

// EmbeddingServiceType represents the type of embedding service to use
type EmbeddingServiceType string

const (
	// MockService uses mock embeddings for testing
	MockService EmbeddingServiceType = "mock"
	// HTTPService uses HTTP calls to external service
	HTTPService EmbeddingServiceType = "http"
)

// EmbeddingServiceFactory creates embedding services based on configuration
type EmbeddingServiceFactory struct {
	serviceType EmbeddingServiceType
	config      embedding.EmbeddingConfig
	serviceURL  string
}

// NewEmbeddingServiceFactory creates a new factory
func NewEmbeddingServiceFactory(serviceType EmbeddingServiceType, config embedding.EmbeddingConfig, serviceURL string) *EmbeddingServiceFactory {
	return &EmbeddingServiceFactory{
		serviceType: serviceType,
		config:      config,
		serviceURL:  serviceURL,
	}
}

// CreateEmbeddingService creates an embedding service based on the configured type
func (f *EmbeddingServiceFactory) CreateEmbeddingService() (embedding.EmbeddingService, error) {
	logrus.WithField("service_type", f.serviceType).Info("Creating embedding service")

	switch f.serviceType {
	case MockService:
		return f.createMockService()
	case HTTPService:
		return f.createHTTPService()
	default:
		return nil, fmt.Errorf("unsupported embedding service type: %s", f.serviceType)
	}
}

// createMockService creates a mock embedding service for testing
func (f *EmbeddingServiceFactory) createMockService() (embedding.EmbeddingService, error) {
	logrus.Info("Creating mock embedding service")
	return NewMockEmbeddingService(f.config)
}

// createHTTPService creates an HTTP-based embedding service
func (f *EmbeddingServiceFactory) createHTTPService() (embedding.EmbeddingService, error) {
	logrus.Info("Creating HTTP embedding service")
	if f.serviceURL == "" {
		return nil, fmt.Errorf("service URL is required for HTTP embedding service")
	}
	return NewHTTPEmbeddingService(f.config, f.serviceURL)
}

// GetSupportedTypes returns all supported embedding service types
func GetSupportedTypes() []EmbeddingServiceType {
	return []EmbeddingServiceType{
		MockService,
		HTTPService,
	}
}

// IsValidType checks if the given service type is valid
func IsValidType(serviceType EmbeddingServiceType) bool {
	for _, validType := range GetSupportedTypes() {
		if validType == serviceType {
			return true
		}
	}
	return false
}

// GetDefaultType returns the default embedding service type
func GetDefaultType() EmbeddingServiceType {
	return HTTPService // Default to HTTP for simplicity
}
