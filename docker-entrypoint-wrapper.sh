#!/bin/bash

# Add random delay to avoid migration race conditions
if [ "$SERVICE_NAME" = "web" ]; then
    # Only web service should run migrations
    SHOULD_MIGRATE=true
    sleep 2
elif [ "$SERVICE_NAME" = "celery" ]; then
    SHOULD_MIGRATE=false
    sleep 10
elif [ "$SERVICE_NAME" = "celery-beat" ]; then
    SHOULD_MIGRATE=false
    sleep 15
else
    # Default for any other service
    SHOULD_MIGRATE=false
    sleep 5
fi

export SHOULD_MIGRATE

# Ensure BASE_DIR is set
export BASE_DIR=/app

# Check if ENVIRONMENT variable is set to 'production'
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Running in production mode with entrypoint.prod.sh and .env.prod"
    # Use a safer method to export variables
    set -a
    source /app/.env.prod
    set +a
    exec /app/entrypoint.prod.sh "$@"
else
    echo "Running in development mode with entrypoint.sh and .env"
    # Use a safer method to export variables
    set -a
    source /app/.env
    set +a
    exec /app/entrypoint.sh "$@"
fi
