#!/bin/bash

# Check if ENVIRONMENT variable is set to 'production'
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Running in production mode with entrypoint.prod.sh and .env.prod"
    # Export all variables from .env.prod to environment
    export $(grep -v '^#' /app/.env.prod | xargs)
    exec /app/entrypoint.prod.sh "$@"
else
    echo "Running in development mode with entrypoint.sh and .env"
    # Export all variables from .env to environment
    export $(grep -v '^#' /app/.env | xargs)
    exec /app/entrypoint.sh "$@"
fi
