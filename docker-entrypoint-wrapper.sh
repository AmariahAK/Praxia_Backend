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
    # Longer wait times for production
    if [ "$ENVIRONMENT" = "production" ]; then
        sleep 30
    else
        sleep 20
    fi
elif [ "$SERVICE_NAME" = "celery-beat" ]; then
    SHOULD_MIGRATE=false
    echo "Celery-beat service: Waiting for web service to be ready"
    # Longer wait times for production
    if [ "$ENVIRONMENT" = "production" ]; then
        sleep 35
    else
        sleep 25
    fi
elif [ "$SERVICE_NAME" = "celery-monitor" ]; then
    SHOULD_MIGRATE=false
    echo "Celery-monitor service: Waiting for web service to be ready"
    # Longer wait times for production
    if [ "$ENVIRONMENT" = "production" ]; then
        sleep 40
    else
        sleep 30
    fi
else
    # Default for any other service
    SHOULD_MIGRATE=false
    echo "Other service: Skipping migrations"
    if [ "$ENVIRONMENT" = "production" ]; then
        sleep 10
    else
        sleep 5
    fi
fi

export SHOULD_MIGRATE

# Ensure BASE_DIR is set
export BASE_DIR=/app

# Create logs directory if it doesn't exist
mkdir -p /app/logs

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
        echo "Loaded SERVICE_NAME: $SERVICE_NAME"
        echo "Loaded ENVIRONMENT: $ENVIRONMENT"
    else
        echo "Warning: .env.prod not found"
    fi
    
    # Additional production-specific environment setup
    export DJANGO_SETTINGS_MODULE=praxia_backend.settings
    export PYTHONPATH=/app:$PYTHONPATH
    
    # Set production-specific resource limits
    if [ "$SERVICE_NAME" = "celery" ]; then
        export CELERY_WORKER_MAX_MEMORY_PER_CHILD=600000
        export CELERY_WORKER_MAX_TASKS_PER_CHILD=100
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
        echo "Loaded SERVICE_NAME: $SERVICE_NAME"
        echo "Loaded ENVIRONMENT: ${ENVIRONMENT:-development}"
    else
        echo "Warning: .env not found"
    fi
    
    # Development-specific environment setup
    export DJANGO_SETTINGS_MODULE=praxia_backend.settings
    export PYTHONPATH=/app:$PYTHONPATH
    
    exec /app/entrypoint.sh "$@"
fi
