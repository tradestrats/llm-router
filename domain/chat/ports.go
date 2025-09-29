package chat

import "context"

// ProviderPort abstracts a chat provider (e.g., OpenRouter)
type ProviderPort interface {
	// Non-streaming chat
	Chat(ctx context.Context, req *Request) (*Response, error)
}

// StreamHandler is a generic callback for streaming chunks
type StreamHandler[T any] func(chunk T) error

// StreamProviderPort supports streaming
type StreamProviderPort[T any] interface {
	Stream(ctx context.Context, req *Request, onChunk StreamHandler[T]) error
}
