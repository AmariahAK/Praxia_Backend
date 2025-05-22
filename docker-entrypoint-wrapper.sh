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
    # Use grep to extract variable assignments and properly export them
    grep -v '^#' /app/.env.prod | while IFS= read -r line; do
        if [[ $line =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
            export "$line"
        fi
    done
    exec /app/entrypoint.prod.sh "$@"
else
    echo "Running in development mode with entrypoint.sh and .env"
    # Use grep to extract variable assignments and properly export them
    grep -v '^#' /app/.env | while IFS= read -r line; do
        if [[ $line =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
            export "$line"
        fi
    done
    exec /app/entrypoint.sh "$@"
fi
