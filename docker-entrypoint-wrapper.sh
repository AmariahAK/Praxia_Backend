#!/bin/bash

# Add random delay to avoid migration race conditions
if [ "$SERVICE_NAME" = "web" ]; then
    # Only web service should run migrations
    SHOULD_MIGRATE=true
    echo "Web service: Will handle migrations"
    sleep 2
elif [ "$SERVICE_NAME" = "celery" ]; then
    SHOULD_MIGRATE=false
    echo "Celery service: Waiting for web service to be ready"
    sleep 20  # Increased wait time
elif [ "$SERVICE_NAME" = "celery-beat" ]; then
    SHOULD_MIGRATE=false
    echo "Celery-beat service: Waiting for web service to be ready"
    sleep 25  # Increased wait time
elif [ "$SERVICE_NAME" = "celery-monitor" ]; then
    SHOULD_MIGRATE=false
    echo "Celery-monitor service: Waiting for web service to be ready"
    sleep 30  # Increased wait time
else
    # Default for any other service
    SHOULD_MIGRATE=false
    echo "Other service: Skipping migrations"
    sleep 5
fi

export SHOULD_MIGRATE

# Ensure BASE_DIR is set
export BASE_DIR=/app

# Check if ENVIRONMENT variable is set to 'production'
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Running in production mode with entrypoint.prod.sh and .env.prod"
    # Load .env.prod and export all variables
    if [ -f /app/.env.prod ]; then
        echo "Loading environment variables from .env.prod"
        set -a
        source /app/.env.prod
        set +a
        echo "Loaded DB_NAME: $DB_NAME"
        echo "Loaded DB_HOST: $DB_HOST"
    else
        echo "Warning: .env.prod not found"
    fi
    exec /app/entrypoint.prod.sh "$@"
else
    echo "Running in development mode with entrypoint.sh and .env"
    # Load .env and export all variables
    if [ -f /app/.env ]; then
        echo "Loading environment variables from .env"
        set -a
        source /app/.env
        set +a
        echo "Loaded DB_NAME: $DB_NAME"
        echo "Loaded DB_HOST: $DB_HOST"
    else
        echo "Warning: .env not found"
    fi
    exec /app/entrypoint.sh "$@"
fi
