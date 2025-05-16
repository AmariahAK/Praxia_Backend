#!/bin/bash

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

# Create database if it doesn't exist
echo "Checking if database exists..."
if ! psql -h $DB_HOST -p $DB_PORT -U $DB_USER -lqt | cut -d \| -f 1 | grep -qw $DB_NAME; then
    echo "Creating database $DB_NAME..."
    createdb -h $DB_HOST -p $DB_PORT -U $DB_USER $DB_NAME
    echo "Database $DB_NAME created successfully!"
else
    echo "Database $DB_NAME already exists."
fi

# Download model weights if they don't exist
echo "Checking for DenseNet121 model weights..."
if [ ! -f "/app/data/models/densenet_xray.pth" ]; then
  echo "Downloading DenseNet121 model weights..."
  python -m api.utils.download_model
  echo "Model weights downloaded successfully!"
else
  echo "Model weights already exist, skipping download."
fi

# Apply database migrations
echo "Applying database migrations..."
python manage.py migrate

# Create superuser if needed
echo "Creating superuser if needed..."
if [ "$DJANGO_SUPERUSER_USERNAME" ] && [ "$DJANGO_SUPERUSER_EMAIL" ] && [ "$DJANGO_SUPERUSER_PASSWORD" ]; then
  python manage.py createsuperuser --noinput --username $DJANGO_SUPERUSER_USERNAME --email $DJANGO_SUPERUSER_EMAIL || true
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
  cat > /app/prometheus/prometheus.yml << EOF
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

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
EOF
fi

# Run health check
echo "Running initial health check..."
python manage.py shell -c "from api.AI.ai_healthcheck import startup_health_check; startup_health_check()"

# Start server with Daphne
echo "Starting Daphne server..."
exec daphne -b 0.0.0.0 -p 8000 praxia_backend.asgi:application
