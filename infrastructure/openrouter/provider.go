package openrouter

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"sync"
	"time"

	appchat "llm-router/domain/chat"
	"llm-router/domain/bandit"

	"github.com/sirupsen/logrus"
)

type Provider struct {
	apiKey           string
	baseURL          string
	models           []string
	httpClient       *http.Client
	rng              *rand.Rand
	rngMutex         sync.Mutex
	refererURL       string
	appName          string
	contextualRouter bandit.SimilarityRouter // Optional contextual router
}

func NewProvider(apiKey, baseURL string, models []string, refererURL, appName string) *Provider {
	// Configure HTTP client with connection pooling
	transport := &http.Transport{
		MaxIdleConns:          200,
		MaxIdleConnsPerHost:   100,
		MaxConnsPerHost:       200,
		IdleConnTimeout:       90 * time.Second,
		DisableCompression:    false,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		ResponseHeaderTimeout: 30 * time.Second,
	}

	return &Provider{
		apiKey:  apiKey,
		baseURL: baseURL,
		models:  models,
		httpClient: &http.Client{
			Timeout:   60 * time.Second,
			Transport: transport,
		},
		rng:        rand.New(rand.NewSource(time.Now().UnixNano())),
		refererURL: refererURL,
		appName:    appName,
	}
}

// SetContextualRouter sets the contextual router for intelligent model selection
func (p *Provider) SetContextualRouter(router bandit.SimilarityRouter) {
	p.contextualRouter = router
}

// selectModel selects a model using contextual routing if available, otherwise falls back to random
func (p *Provider) selectModel(ctx context.Context, req *appchat.Request) string {
	// If user specified a model and it's in our allowed list, use it
	if req.Model != "" && p.isModelAllowed(req.Model) {
		logrus.WithField("model", req.Model).Debug("Using user-specified model")
		return req.Model
	} else if req.Model != "" {
		logrus.WithFields(logrus.Fields{
			"requested_model": req.Model,
			"allowed_models":  p.models,
		}).Warn("Requested model not in allowed list, using intelligent routing")
	}

	// Try contextual routing first
	if p.contextualRouter != nil {
		if model, err := p.contextualRouter.SelectModel(ctx, req); err == nil && model != "" {
			logrus.WithField("model", model).Debug("Selected model using contextual router")
			return model
		} else if err != nil {
			logrus.WithError(err).Debug("Contextual router failed, falling back to random selection")
		}
	}

	// Fallback to random selection
	return p.getRandomModel()
}

// isModelAllowed checks if a model is in the allowed models list
func (p *Provider) isModelAllowed(model string) bool {
	for _, allowedModel := range p.models {
		if allowedModel == model {
			return true
		}
	}
	return false
}

func (p *Provider) getRandomModel() string {
	if len(p.models) == 0 {
		return ""
	}
	p.rngMutex.Lock()
	defer p.rngMutex.Unlock()
	return p.models[p.rng.Intn(len(p.models))]
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type usageOptions struct {
	Include bool `json:"include"`
}

type apiChatRequest struct {
	Model         string            `json:"model"`
	Messages      []appchat.Message `json:"messages"`
	Stream        bool              `json:"stream,omitempty"`
	StreamOptions *streamOptions    `json:"stream_options,omitempty"`
	Usage         *usageOptions     `json:"usage,omitempty"`
}

func (p *Provider) Chat(ctx context.Context, req *appchat.Request) (*appchat.Response, error) {
	return p.chatWithRetry(ctx, req, 3)
}

func (p *Provider) chatWithRetry(ctx context.Context, req *appchat.Request, maxRetries int) (*appchat.Response, error) {
	var lastErr error

	for attempt := 0; attempt < maxRetries; attempt++ {
		if attempt > 0 {
			// Exponential backoff: 1s, 2s, 4s
			base := time.Duration(math.Pow(2, float64(attempt-1))) * time.Second
			// Add simple jitter of up to 250ms
			jitter := time.Duration(p.rng.Intn(250)) * time.Millisecond
			backoff := base + jitter
			logrus.WithFields(logrus.Fields{
				"attempt": attempt + 1,
				"backoff": backoff,
			}).Info("Retrying API call after backoff")
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		model := p.selectModel(ctx, req)
		if model == "" {
			return nil, fmt.Errorf("no models available")
		}

		jsonData, err := json.Marshal(apiChatRequest{
			Model:    model,
			Messages: req.Messages,
			Stream:   false,
			Usage:    &usageOptions{Include: true},
		})
		if err != nil {
			return nil, fmt.Errorf("marshal: %w", err)
		}

		hreq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/chat/completions", bytes.NewBuffer(jsonData))
		if err != nil {
			return nil, fmt.Errorf("new request: %w", err)
		}
		hreq.Header.Set("Content-Type", "application/json")
		hreq.Header.Set("Authorization", "Bearer "+p.apiKey)
		hreq.Header.Set("HTTP-Referer", p.refererURL)
		hreq.Header.Set("X-Title", p.appName)

		resp, err := p.httpClient.Do(hreq)
		if err != nil {
			lastErr = fmt.Errorf("do: %w", err)
			continue
		}

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			// Ensure body is closed before retry
			resp.Body.Close()
			lastErr = fmt.Errorf("read: %w", err)
			continue
		}

		// Retry on server errors (5xx) or rate limiting (429)
		if resp.StatusCode >= 500 || resp.StatusCode == 429 {
			// Close before next retry to avoid leaking connections
			resp.Body.Close()
			lastErr = fmt.Errorf("openrouter api error: status %d, model %s: %s", resp.StatusCode, model, string(body))
			logrus.WithFields(logrus.Fields{"status": resp.StatusCode, "body": string(body), "model": model, "attempt": attempt + 1}).Warn("Retryable API error")
			continue
		}

		if resp.StatusCode != http.StatusOK {
			// Non-retryable error
			logrus.WithFields(logrus.Fields{"status": resp.StatusCode, "body": string(body), "model": model}).Error("OpenRouter API error")
			// Close before returning
			resp.Body.Close()
			return nil, fmt.Errorf("openrouter api error: status %d, model %s: %s", resp.StatusCode, model, string(body))
		}

		var out appchat.Response
		if err := json.Unmarshal(body, &out); err != nil {
			// Close before retrying
			resp.Body.Close()
			lastErr = fmt.Errorf("unmarshal: %w", err)
			continue
		}

		// Successful response — close the body and return
		resp.Body.Close()
		return &out, nil
	}

	return nil, fmt.Errorf("api call failed after %d attempts: %w", maxRetries, lastErr)
}

func (p *Provider) Stream(ctx context.Context, req *appchat.Request, onChunk appchat.StreamHandler[appchat.StreamChunk]) error {
	model := p.selectModel(ctx, req)
	if model == "" {
		return fmt.Errorf("no models available")
	}

	jsonData, err := json.Marshal(apiChatRequest{
		Model:         model,
		Messages:      req.Messages,
		Stream:        true,
		StreamOptions: &streamOptions{IncludeUsage: true},
		Usage:         &usageOptions{Include: true},
	})
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}

	hreq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("new request: %w", err)
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Authorization", "Bearer "+p.apiKey)
	hreq.Header.Set("HTTP-Referer", p.refererURL)
	hreq.Header.Set("X-Title", p.appName)

	resp, err := p.httpClient.Do(hreq)
	if err != nil {
		return fmt.Errorf("do: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		logrus.WithFields(logrus.Fields{"status": resp.StatusCode, "body": string(body), "model": model}).Error("OpenRouter streaming API error")
		return fmt.Errorf("openrouter streaming api error: status %d, model %s: %s", resp.StatusCode, model, string(body))
	}

	reader := bufio.NewReader(resp.Body)
	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return fmt.Errorf("stream read: %w", err)
		}
		if len(line) < 6 || string(line[:6]) != "data: " {
			continue
		}
		payload := bytes.TrimSpace(line[6:])
		if bytes.Equal(payload, []byte("[DONE]")) {
			return nil
		}
		var chunk appchat.StreamChunk
		if err := json.Unmarshal(payload, &chunk); err != nil {
			logrus.WithFields(logrus.Fields{"payload": string(payload), "model": model}).Error("Failed to decode streaming chunk")
			return fmt.Errorf("decode chunk for model %s: %w", model, err)
		}
		if err := onChunk(chunk); err != nil {
			return err
		}
	}
}
