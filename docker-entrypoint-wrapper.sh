#!/bin/bash

# Add random delay to avoid migration race conditions
if [ "$SERVICE_NAME" = "celery" ]; then
    sleep 5
elif [ "$SERVICE_NAME" = "celery-beat" ]; then
    sleep 10
fi

# Check if ENVIRONMENT variable is set to 'production'
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Running in production mode with entrypoint.prod.sh and .env.prod"
    # Export all variables from .env.prod to environment
    set -a
    source /app/.env.prod
    set +a
    exec /app/entrypoint.prod.sh "$@"
else
    echo "Running in development mode with entrypoint.sh and .env"
    # Export all variables from .env to environment
    set -a
    source /app/.env
    set +a
    exec /app/entrypoint.sh "$@"
fi
