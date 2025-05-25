#!/bin/bash

# Set up error handling
set -e

echo "=== Starting ${SERVICE_NAME:-unknown} service ==="

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1
    
    echo "Waiting for $service_name to be ready..."
    while ! nc -z $host $port; do
        if [ $attempt -eq $max_attempts ]; then
            echo "ERROR: $service_name not ready after $max_attempts attempts"
            exit 1
        fi
        echo "Attempt $attempt/$max_attempts: $service_name not ready, waiting..."
        sleep 2
        attempt=$((attempt + 1))
    done
    echo "$service_name is ready!"
}

# Wait for Redis first (all services need it)
wait_for_service redis 6379 "Redis"

# Only download model for web service or if explicitly needed
if [ "$SERVICE_NAME" = "web" ] || [ "$INITIALIZE_XRAY_MODEL" = "True" ]; then
    echo "Checking for DenseNet121 model weights..."
    if [ ! -f "/app/data/models/densenet_xray.pth" ] && [ ! -f "/app/data/models/densenet_xray_fixed.pth" ]; then
        echo "Downloading DenseNet121 model weights..."
        for i in {1..3}; do
            echo "Attempt $i to download model weights..."
            if python -m api.utils.download_model; then
                echo "Model weights downloaded successfully!"
                break
            else
                echo "Download failed, retrying..."
                sleep 10
            fi
        done
        
        if [ ! -f "/app/data/models/densenet_xray.pth" ]; then
            echo "Warning: Failed to download model weights after multiple attempts."
            touch /app/data/models/densenet_xray.pth.failed
        fi
    else
        echo "Model weights already exist"
    fi
    
    # Fix model if needed
    if [ -f "/app/data/models/densenet_xray.pth" ] && [ ! -f "/app/data/models/densenet_xray_fixed.pth" ]; then
        echo "Fixing model architecture..."
        python -c "
try:
    from api.utils.model_fix import fix_densenet_model
    fix_densenet_model()
    print('Model fixed successfully')
except Exception as e:
    print(f'Error fixing model: {e}')
" || echo "Model fix failed, continuing..."
    fi
fi

# Apply database migrations only for web service
if [ "$SHOULD_MIGRATE" = "true" ]; then
    echo "Applying database migrations..."
    python manage.py migrate --noinput
    
    # Create superuser only once
    echo "Creating superuser if needed..."
    if [ "$DJANGO_SUPERUSER_USERNAME" ] && [ "$DJANGO_SUPERUSER_EMAIL" ] && [ "$DJANGO_SUPERUSER_PASSWORD" ]; then
        python manage.py shell -c "
from django.contrib.auth import get_user_model
User = get_user_model()
try:
    if not User.objects.filter(username='$DJANGO_SUPERUSER_USERNAME').exists():
        User.objects.create_superuser('$DJANGO_SUPERUSER_USERNAME', '$DJANGO_SUPERUSER_EMAIL', '$DJANGO_SUPERUSER_PASSWORD')
        print('Superuser created successfully')
    else:
        print('Superuser already exists')
except Exception as e:
    print(f'Error creating superuser: {e}')
" || echo "Superuser creation failed, continuing..."
    fi
    
    # Collect static files
    echo "Collecting static files..."
    python manage.py collectstatic --noinput
    
    # Run health check
    echo "Running initial health check..."
    python manage.py shell -c "
try:
    from api.AI.ai_healthcheck import startup_health_check
    result = startup_health_check()
    print('Health check completed successfully')
except Exception as e:
    print(f'Health check failed: {e}')
" || echo "Health check failed, continuing..."
    
else
    echo "Skipping database migrations for this service..."
fi

# For non-web services, wait for web service to be ready
if [ "$SERVICE_NAME" != "web" ]; then
    echo "Waiting for web service to be ready..."
    max_attempts=60
    attempt=1
    while ! curl -f http://web:8000/api/health/ >/dev/null 2>&1; do
        if [ $attempt -eq $max_attempts ]; then
            echo "WARNING: Web service not ready after $max_attempts attempts, continuing anyway..."
            break
        fi
        echo "Attempt $attempt/$max_attempts: Web service not ready, waiting..."
        sleep 5
        attempt=$((attempt + 1))
    done
    echo "Web service is ready (or timeout reached)!"
fi

# Create necessary directories
mkdir -p /app/media/profile_pics /app/media/xrays /app/data/models /app/prometheus

# Clear Celery queues for celery services
if [[ "$SERVICE_NAME" == celery* ]]; then
    echo "Clearing Celery broker database..."
    python -c "
import redis
try:
    r = redis.Redis(host='redis', port=6379, db=1)
    r.flushdb()
    print('Cleared Celery broker database')
except Exception as e:
    print(f'Could not clear Celery database: {e}')
" || echo "Failed to clear Celery database, continuing..."
fi

echo "=== Starting Daphne server for ${SERVICE_NAME:-unknown} ==="
exec daphne -b 0.0.0.0 -p 8000 praxia_backend.asgi:application
