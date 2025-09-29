package chat

// Core chat entities independent of frameworks and vendors

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Request struct {
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream,omitempty"`
	Model    string    `json:"model,omitempty"`
}

type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

type Usage struct {
	PromptTokens     int          `json:"prompt_tokens"`
	CompletionTokens int          `json:"completion_tokens"`
	TotalTokens      int          `json:"total_tokens"`
	Cost             *float64     `json:"cost,omitempty"`
	CostDetails      *CostDetails `json:"cost_details,omitempty"`
}

type CostDetails struct {
	UpstreamInferenceCost *float64 `json:"upstream_inference_cost,omitempty"`
}

type Response struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// Streaming chunk types (OpenAI/OpenRouter-compatible)
type StreamDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type StreamChoice struct {
	Index        int         `json:"index"`
	Delta        StreamDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason,omitempty"`
}

type StreamChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
	Usage   *Usage         `json:"usage,omitempty"`
}

type ErrorResponse struct {
	Error string `json:"error"`
}
