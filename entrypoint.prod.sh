#!/bin/bash

# Download model weights if they don't exist
echo "Checking for DenseNet121 model weights..."
if [ ! -f "/app/data/models/densenet_xray.pth" ]; then
  echo "Downloading DenseNet121 model weights..."
  # Try up to 3 times to download the model
  for i in {1..3}; do
    echo "Attempt $i to download model weights..."
    python -m api.utils.download_model && break || echo "Download failed, retrying..."
    sleep 5
  done
  
  if [ ! -f "/app/data/models/densenet_xray.pth" ]; then
    echo "Warning: Failed to download model weights after multiple attempts."
    # Create a placeholder file to prevent repeated download attempts
    touch /app/data/models/densenet_xray.pth.failed
  else
    echo "Model weights downloaded successfully!"
  fi
else
  echo "Model weights already exist, skipping download."
fi

# Apply database migrations only if this service should migrate
if [ "$SHOULD_MIGRATE" = "true" ]; then
  echo "Applying database migrations..."
  python manage.py migrate
else
  echo "Skipping database migrations for this service..."
fi

# Create superuser if needed
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
"
fi

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --no-input

# Create necessary directories
mkdir -p /app/media/profile_pics
mkdir -p /app/media/xrays
mkdir -p /app/data/models
mkdir -p /app/prometheus

# Create Prometheus config if it doesn't exist
if [ ! -f /app/prometheus/prometheus.yml ]; then
  echo "Creating Prometheus config..."
  cat > /app/prometheus/prometheus.yml << EOC
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'praxia'
    scrape_interval: 5s
    static_configs:
      - targets: ['web:8000']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
EOC
fi

# Run health check
echo "Running initial health check..."
python manage.py shell -c "from api.AI.ai_healthcheck import startup_health_check; startup_health_check()" || echo "Health check failed, continuing..."

# Start server with Daphne
echo "Starting Daphne server..."
exec daphne -b 0.0.0.0 -p 8000 praxia_backend.asgi:application
