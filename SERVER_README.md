# BitNet LLM HTTP Server

This project provides an HTTP server for serving the BitNet LLM model via a RESTful API. The server is packaged as a Docker container for easy deployment.

## Features

- RESTful API for text generation with BitNet LLM
- Customizable system prompt
- Adjustable generation parameters (temperature, max tokens, etc.)
- Health check endpoint
- Model information endpoint
- Persistent model processes for faster response times
- Process pool for handling concurrent requests

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
docker-compose up -d
```

This will build the image and start the server in detached mode. The server will be available at http://localhost:8000.

To stop the server:

```bash
docker-compose down
```

### Option 2: Using Docker Directly

#### Building the Docker Image

```bash
docker build -t bitnet-llm-server .
```

#### Running the Server

```bash
docker run -p 8000:8000 bitnet-llm-server
```

The server will be available at http://localhost:8000

## Environment Variables

You can configure the server using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| MAX_THREADS | Maximum number of threads to use for inference | 4 |
| MODEL_INSTANCES | Number of persistent model processes to maintain | 1 |

To set environment variables with Docker Compose, edit the `docker-compose.yml` file.

## How It Works

The server maintains a pool of persistent llama-cli processes that stay alive between requests. When a request comes in:

1. The server selects the next available process from the pool (round-robin)
2. The request is sent to the process via its stdin
3. The response is read from the process's stdout
4. The response is parsed and returned to the client

This approach eliminates the overhead of starting a new process for each request, resulting in much faster response times.

## API Usage

### Generate Text

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning?",
    "system_prompt": "You are a helpful AI assistant that provides clear and concise explanations about technical topics.",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### Health Check

```bash
curl "http://localhost:8000/health"
```

### Model Information

```bash
curl "http://localhost:8000/model_info"
```

## API Parameters

The `/generate` endpoint accepts the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| prompt | string | (required) | The user prompt for text generation |
| system_prompt | string | "You are a helpful, harmless, and honest AI assistant. Answer the user's questions truthfully and helpfully." | System prompt that controls the AI's behavior |
| max_tokens | integer | 256 | Maximum number of tokens to generate |
| temperature | float | 0.7 | Controls randomness (higher = more random) |
| top_p | float | 0.9 | Controls diversity via nucleus sampling |
| context_size | integer | 2048 | Size of the context window |
| threads | integer | (from MAX_THREADS env var) | Number of CPU threads to use |
| conversation_mode | boolean | true | Whether to use conversation mode |

## Response Format

The API returns a JSON object with the following structure:

```json
{
  "generated_text": "Machine learning is a subset of artificial intelligence...",
  "tokens_generated": 45,
  "tokens_processed": 15
}
```

## Performance Considerations

- **MODEL_INSTANCES**: Increasing this value creates more processes, allowing more concurrent requests at the cost of higher memory usage. For high-traffic servers, consider setting this to 2-4.
- **MAX_THREADS**: Higher values improve single-request performance but consume more CPU resources. Finding the right balance depends on your server's CPU capacity.

## Requirements

- Docker
- Docker Compose (for the recommended setup)
- Internet connection (for building the image and downloading the model)

## Data Persistence

The Docker Compose configuration includes a volume mount for the models directory, ensuring that model data persists across container restarts. 