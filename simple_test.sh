#!/bin/bash

# Default values
HOST=${HOST:-"localhost"}
PORT=${PORT:-"8000"}
PROMPT=${PROMPT:-"What is machine learning?"}

# Check if prompt was provided as an argument
if [ $# -gt 0 ]; then
    PROMPT="$1"
fi

echo "Sending request to http://$HOST:$PORT/generate"
echo "Prompt: $PROMPT"
echo "---------------------------------------------"

# Send the request
start_time=$(date +%s)

curl -X POST "http://$HOST:$PORT/health"

curl -X POST "http://$HOST:$PORT/generate" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"$PROMPT\"}"

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo ""
echo "---------------------------------------------"
echo "Request completed in $elapsed seconds" 