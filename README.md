# LLM Router

Intelligent LLM router using contextual bandits and semantic similarity. Automatically learns which models perform best for different types of requests without an offline training loop. Benchmarks coming soon, and there are interesting knobs to tune for your use case.

## Quick Start

1. **Setup configuration:**
   ```bash
   cp config.env .env
   # Edit .env and add your OPENROUTER_API_KEY
   ```

2. **Start services:**
   ```bash
   docker-compose up -d
   ```

3. **View logs:**
   ```bash
   docker-compose logs -f llm-proxy
   ```

4. **Stop services:**
   ```bash
   docker-compose down
   ```

## How It Works

LLM Router uses a **contextual bandit algorithm** with Thompson sampling to intelligently route requests to the best-performing model based on:

- **Semantic similarity** to past successful requests
- **Real-time feedback** from user ratings and response quality
- **Performance metrics** including latency and cost
- **Exploration vs exploitation** balance to discover better models

The system starts with reasonable defaults and learns from every interaction, gradually improving model selection for your specific use cases.

## Essential Settings

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `OPENROUTER_API_KEY` | **Yes** | Your OpenRouter API key | - |
| `LLM_MODELS` | No | Comma-separated model list | Free models included |
| `PORT` | No | Server port | 8080 |
| `LOG_LEVEL` | No | Logging level (debug/info/warn/error) | info |
| `DEFAULT_MODEL` | No | Fallback model when bandit is uncertain | claude-3.5-sonnet |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/completions` | POST | Chat completions (requires X-Request-ID header) |
| `/feedback` | POST | Submit user feedback for requests |
| `/health` | GET | Health status of all services |
| `/ready` | GET | Readiness check |
| `/metrics/{request-id}` | GET | Get metrics for specific request |
| `/requests/{request-id}` | GET | Get complete request details |

### Chat Request Example

```bash
curl -X POST http://localhost:8080/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: $(uuidgen)" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### Feedback Example

```bash
curl -X POST http://localhost:8080/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "your-request-id",
    "score": 0.8,
    "feedback_text": "Good response"
  }'
```

## Configuration Guide

### Tuning the Bandit Algorithm

The contextual bandit can be tuned for your specific use case. Here are the key parameters and when to adjust them:

#### Thompson Sampling Weights

```bash
# In your .env file:
FEEDBACK_WEIGHT=0.6      # How much user feedback influences selection (0-1)
LATENCY_WEIGHT=0.2       # How much response speed matters (positive = faster better)
COST_WEIGHT=-0.2         # How much cost matters (negative = cheaper better)
EXPLORATION_RATE=0.1     # How often to try non-optimal models (0-1)
```

**When to adjust:**
- **High feedback weight (0.7-0.9)**: When you have reliable user feedback
- **High latency weight (0.3-0.5)**: For real-time applications
- **High cost weight (-0.5)**: When minimizing costs is critical
- **High exploration (0.2-0.3)**: When you want faster learning but less optimal short-term performance

#### Similarity Matching

```bash
SIMILARITY_THRESHOLD=0.7     # How similar requests must be to reuse learnings (0-1)
MAX_SIMILAR_REQUESTS=50      # Max historical requests to consider
RECENCY_DAYS=30             # Only consider requests from last N days
MIN_SIMILAR_REQUESTS=5       # Min similar requests needed to use similarity
```

**When to adjust:**
- **Lower threshold (0.5-0.6)**: More aggressive similarity matching, faster learning
- **Higher threshold (0.8-0.9)**: More conservative, only very similar requests
- **More max requests (100+)**: Better decisions but slower queries
- **Shorter recency (7-14 days)**: Adapt quickly to changing preferences

#### Cold Start Behavior

```bash
MIN_CONFIDENCE_SCORE=0.1     # Minimum confidence before using bandit
OPTIMISTIC_PRIOR=0.8         # Starting assumption about model quality (0-1)
EXPLORATION_BONUS=0.1        # Extra reward for trying new models
MIN_REQUESTS_FOR_GLOBAL=10   # Requests needed before using global stats
```

### Environment Variables Reference

| Category | Variable | Type | Default | Description |
|----------|----------|------|---------|-------------|
| **Server** | `HOST` | string | 0.0.0.0 | Server bind address |
| | `PORT` | string | 8080 | Server port |
| | `CORS_ORIGINS` | string | * | Comma-separated CORS origins |
| **Models** | `OPENROUTER_API_KEY` | string | **required** | Your OpenRouter API key |
| | `LLM_MODELS` | string | free models | Comma-separated model list |
| | `DEFAULT_MODEL` | string | claude-3.5-sonnet | Fallback model |
| **Bandit** | `FEEDBACK_WEIGHT` | float | 0.6 | User feedback influence (0-1) |
| | `LATENCY_WEIGHT` | float | 0.2 | Response speed importance |
| | `COST_WEIGHT` | float | -0.2 | Cost importance (negative) |
| | `EXPLORATION_RATE` | float | 0.1 | Exploration vs exploitation (0-1) |
| | `SIMILARITY_THRESHOLD` | float | 0.7 | Similarity matching threshold (0-1) |
| | `MAX_SIMILAR_REQUESTS` | int | 50 | Max historical requests to consider |
| **Database** | `ENABLE_PERSISTENCE` | bool | false | Enable request/feedback storage |
| | `DATABASE_URL` | string | - | Full database connection string |
| | `DATABASE_HOST` | string | localhost | Database host |
| | `DATABASE_PORT` | string | 5432 | Database port |
| **Logging** | `LOG_LEVEL` | string | info | debug/info/warn/error |
| | `LOG_FORMAT` | string | auto | json/text/auto |

### Example Configurations

Pre-configured YAML files are available for different environments:

#### Development Setup (Fast Learning)
```bash
# Use the development config
docker-compose up -d --env-file config.dev.yaml

# Or copy the example environment file
cp .env.example .env
# Edit .env with your API key
```

The `config.dev.yaml` includes:
- High exploration rate (0.3) for faster learning
- Debug logging for development
- Lower similarity threshold for aggressive matching
- Only free models to avoid unexpected costs

#### Production Setup (Optimized Performance)
```bash
# Use the production config
docker-compose up -d --env-file config.prod.yaml
```

The `config.prod.yaml` includes:
- Low exploration rate (0.05) for stable performance
- Higher latency weight for responsive service
- Database persistence enabled
- SSL required for security
- JSON logging for monitoring

#### Cost-Optimized Setup
```bash
# Use the cost-optimized config
docker-compose up -d --env-file config.cost-optimized.yaml
```

The `config.cost-optimized.yaml` includes:
- Only free models (no paid API calls)
- Very high cost weight (-0.8)
- Moderate exploration among free models
- Optimized for learning quality differences between free models

#### Using Custom Configurations
```bash
# Specify a custom config file
CONFIG_FILE=config.dev.yaml docker-compose up -d

# Or set environment variables directly
OPENROUTER_API_KEY=your_key EXPLORATION_RATE=0.2 docker-compose up -d
```

## Troubleshooting

### Common Issues

**Models not available:**
- Check your `OPENROUTER_API_KEY` is valid
- Verify model names match OpenRouter's format: `provider/model-name`
- Some models require credits or specific API access

**Slow routing decisions:**
- Reduce `MAX_SIMILAR_REQUESTS` for faster queries
- Increase `SIMILARITY_THRESHOLD` to be more selective
- Enable database persistence to avoid recalculating embeddings

**Poor model selection:**
- Increase `EXPLORATION_RATE` to try more models
- Provide feedback via the `/feedback` endpoint
- Lower `SIMILARITY_THRESHOLD` to learn from more examples

**High costs:**
- Increase `COST_WEIGHT` (make it more negative)
- Set `LLM_MODELS` to only include free models
- Use a free model as `DEFAULT_MODEL`

### Health Checks

```bash
# Check if services are running
curl http://localhost:8080/health

# Check if embedding service is ready
curl http://localhost:8080/ready

# View metrics for a specific request
curl http://localhost:8080/metrics/your-request-id
```