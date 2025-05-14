#!/bin/bash

# Set default values if environment variables are not set
DB_HOST=${DB_HOST:-your-production-db-host.example.com}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-praxia_db}
DB_USER=${DB_USER:-praxia_user}

# Wait for database to be ready
echo "Waiting for PostgreSQL at $DB_HOST:$DB_PORT..."
for i in {1..30}; do
  if pg_isready -h $DB_HOST -p $DB_PORT -U $DB_USER; then
    echo "PostgreSQL started"
    break
  fi
  echo "Attempt $i: PostgreSQL not ready yet, waiting..."
  sleep 2
  if [ $i -eq 30 ]; then
    echo "Error: Could not connect to PostgreSQL after 30 attempts. Exiting."
    exit 1
  fi
done

# Apply database migrations
echo "Applying database migrations..."
python manage.py migrate

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --no-input

# Create necessary directories
mkdir -p /app/media/profile_pics
mkdir -p /app/media/xrays
mkdir -p /app/data/models
mkdir -p /app/prometheus

# Run health check
echo "Running initial health check..."
python manage.py shell -c "from api.AI.ai_healthcheck import startup_health_check; startup_health_check()"

# Start server with Daphne
echo "Starting Daphne server..."
exec daphne -b 0.0.0.0 -p 8000 praxia_backend.asgi:application
