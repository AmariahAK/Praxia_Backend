#!/bin/bash

# This script helps initialize the database properly
echo "Initializing database..."

# Connect to the database container
docker-compose exec db psql -U postgres -c "SELECT 1;" || \
docker-compose exec db psql -U $DB_USER -c "CREATE ROLE postgres WITH SUPERUSER LOGIN PASSWORD 'postgres_password';"

# Create the database if it doesn't exist
docker-compose exec db psql -U $DB_USER -c "CREATE DATABASE $DB_NAME;" || echo "Database already exists"

# Grant privileges
docker-compose exec db psql -U $DB_USER -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO postgres;"

echo "Database initialization complete!"
