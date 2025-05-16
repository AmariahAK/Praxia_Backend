#!/bin/bash

echo "Resetting Praxia Backend environment..."

# Stop all containers
docker-compose down

# Remove volumes
docker volume rm praxia_backend_postgres_data || true

# Start fresh
docker-compose up -d db
sleep 10  # Wait for DB to initialize

# Run the DB init script
./scripts/db-init.sh

# Start the rest of the services
docker-compose up -d

echo "Reset complete! The system should now be running with a fresh database."
