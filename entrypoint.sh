#!/bin/bash

# Wait for database to be ready
echo "Waiting for PostgreSQL..."
while ! nc -z $DB_HOST $DB_PORT; do
  sleep 0.1
done
echo "PostgreSQL started"

# Apply database migrations
python manage.py migrate

# Create superuser if needed
python manage.py createsuperuser --noinput || true

# Collect static files
python manage.py collectstatic --no-input

# Start server with Daphne
exec daphne -b 0.0.0.0 -p 8000 praxia_backend.asgi:application
