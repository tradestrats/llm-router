# Build stage
FROM golang:1.23-alpine AS builder

# Install git, ca-certificates, and build dependencies
RUN apk add --no-cache git ca-certificates

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o llm-proxy main.go

# Final stage - Alpine image
FROM alpine:latest

# Install ca-certificates
RUN apk --no-cache add ca-certificates

# Create non-root user
RUN adduser -D -s /bin/sh llm-proxy

# Set working directory
WORKDIR /app

# Copy the binary from builder stage
COPY --from=builder /app/llm-proxy .

# Change ownership to non-root user
RUN chown -R llm-proxy:llm-proxy /app

# Switch to non-root user
USER llm-proxy

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Run the application
CMD ["./llm-proxy"]
