#!/bin/bash

# Set up error handling
set -e

echo "=== Starting ${SERVICE_NAME:-unknown} service (PRODUCTION) ==="

# Ensure environment variables are loaded
if [ -f /app/.env.prod ]; then
    echo "Re-loading environment variables from .env.prod"
    set -a
    source /app/.env.prod
    set +a
    echo "Current DB_NAME: $DB_NAME"
    echo "Current DB_HOST: $DB_HOST"
    echo "Current DB_USER: $DB_USER"
fi

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=60  # Increased for production
    local attempt=1
    
    echo "Waiting for $service_name to be ready..."
    while ! nc -z $host $port; do
        if [ $attempt -eq $max_attempts ]; then
            echo "ERROR: $service_name not ready after $max_attempts attempts"
            exit 1
        fi
        echo "Attempt $attempt/$max_attempts: $service_name not ready, waiting..."
        sleep 3  # Longer wait for production
        attempt=$((attempt + 1))
    done
    echo "$service_name is ready!"
}

# Wait for Redis first (all services need it)
wait_for_service ${REDIS_HOST:-redis} ${REDIS_PORT:-6379} "Redis"

# Only download model for web service or if explicitly needed
if [ "$SERVICE_NAME" = "web" ] || [ "$INITIALIZE_XRAY_MODEL" = "True" ]; then
    echo "Checking for DenseNet121 model weights..."
    if [ ! -f "/app/data/models/densenet_xray.pth" ] && [ ! -f "/app/data/models/densenet_xray_fixed.pth" ]; then
        echo "Downloading DenseNet121 model weights..."
        for i in {1..5}; do  # More attempts in production
            echo "Attempt $i to download model weights..."
            if timeout 300 python -m api.utils.download_model; then  # 5 minute timeout
                echo "Model weights downloaded successfully!"
                break
            else
                echo "Download failed, retrying..."
                sleep 30  # Longer wait between retries
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
        timeout 120 python -c "
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
    echo "Using database: $DB_NAME on host: $DB_HOST"
    
    # Test database connection first
    python -c "
import os
import django
from django.conf import settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'praxia_backend.settings')
django.setup()

from django.db import connections
try:
    db = connections['default']
    cursor = db.cursor()
    cursor.execute('SELECT 1')
    print(f'Database connection successful to: {settings.DATABASES[\"default\"][\"NAME\"]}')
    print(f'Host: {settings.DATABASES[\"default\"][\"HOST\"]}')
    print(f'User: {settings.DATABASES[\"default\"][\"USER\"]}')
except Exception as e:
    print(f'Database connection failed: {e}')
    print(f'Trying to connect to: {settings.DATABASES[\"default\"][\"NAME\"]}')
    print(f'Host: {settings.DATABASES[\"default\"][\"HOST\"]}')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        python manage.py migrate --noinput
        
        # Create superuser only once
        echo "Creating superuser if needed..."
        if [ "$DJANGO_SUPERUSER_USERNAME" ] && [ "$DJANGO_SUPERUSER_EMAIL" ] && [ "$DJANGO_SUPERUSER_PASSWORD" ]; then
            timeout 60 python manage.py shell -c "
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
        python manage.py collectstatic --noinput --clear
        
        # Run health check
        echo "Running initial health check..."
        timeout 120 python manage.py shell -c "
try:
    from api.AI.ai_healthcheck import startup_health_check
    result = startup_health_check()
    print('Health check completed successfully')
except Exception as e:
    print(f'Health check failed: {e}')
" || echo "Health check failed, continuing..."
    else
        echo "Database connection failed, exiting..."
        exit 1
    fi
else
    echo "Skipping database migrations for this service..."
fi

# For non-web services, wait for web service to be ready
if [ "$SERVICE_NAME" != "web" ]; then
    echo "Waiting for web service to be ready..."
    max_attempts=120  # Longer wait in production
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

# For Celery services, ensure they can connect to the broker
if [[ "$SERVICE_NAME" == celery* ]]; then
    echo "Testing Celery broker connection..."
    timeout 30 python -c "
import redis
import sys
try:
    r = redis.Redis(host='${REDIS_HOST:-redis}', port=${REDIS_PORT:-6379}, db=1)
    r.ping()
    print('Celery broker connection successful')
except Exception as e:
    print(f'Celery broker connection failed: {e}')
    sys.exit(1)
" || exit 1

    echo "Clearing Celery broker database..."
    timeout 30 python -c "
import redis
try:
    r = redis.Redis(host='${REDIS_HOST:-redis}', port=${REDIS_PORT:-6379}, db=1)
    r.flushdb()
    print('Cleared Celery broker database')
except Exception as e:
    print(f'Could not clear Celery database: {e}')
" || echo "Failed to clear Celery database, continuing..."
fi

# Create necessary directories
mkdir -p /app/media/profile_pics /app/media/xrays /app/data/models /app/prometheus

# Create Prometheus config if it doesn't exist
if [ ! -f /app/prometheus/prometheus.yml ]; then
    echo "Creating Prometheus config..."
    cat > /app/prometheus/prometheus.yml << EOC
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    environment: 'production'
    cluster: 'praxia-prod'

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'praxia-web'
    scrape_interval: 10s
    static_configs:
      - targets: ['web:8000']
    metrics_path: '/metrics'
    scrape_timeout: 10s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
EOC
fi

# Production-specific optimizations
echo "Applying production optimizations..."

# Set Python optimizations
export PYTHONOPTIMIZE=2
export PYTHONDONTWRITEBYTECODE=1

# Set memory limits for Python
export MALLOC_ARENA_MAX=2

# Warm up the application for web service
if [ "$SERVICE_NAME" = "web" ]; then
    echo "Warming up application..."
    timeout 60 python -c "
try:
    import django
    django.setup()
    from api.AI.praxia_model import PraxiaAI
    # Initialize AI model to warm up
    praxia = PraxiaAI()
    print('Application warmed up successfully')
except Exception as e:
    print(f'Warmup failed: {e}')
" || echo "Application warmup failed, continuing..."
fi

echo "=== Starting Daphne server for ${SERVICE_NAME:-unknown} (PRODUCTION) ==="

# Production server configuration
if [ "$SERVICE_NAME" = "web" ]; then
    # Use Daphne with production settings
    exec daphne -b 0.0.0.0 -p 8000 \
        --proxy-headers \
        --access-log /app/logs/access.log \
        --application-close-timeout 60 \
        --websocket_timeout 86400 \
        praxia_backend.asgi:application
else
    # For other services, use standard Daphne
    exec daphne -b 0.0.0.0 -p 8000 praxia_backend.asgi:application
fi
