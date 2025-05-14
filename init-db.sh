#!/bin/bash
set -e

# Function to create database if it doesn't exist
create_db_if_not_exists() {
    local db_name=$1
    echo "Checking if database $db_name exists..."
    if ! psql -U "$POSTGRES_USER" -lqt | cut -d \| -f 1 | grep -qw "$db_name"; then
        echo "Creating database $db_name..."
        psql -U "$POSTGRES_USER" -c "CREATE DATABASE $db_name;"
        echo "Database $db_name created successfully!"
    else
        echo "Database $db_name already exists."
    fi
}

# Create main database
create_db_if_not_exists "praxia_db"

# Create shard databases
create_db_if_not_exists "praxia_shard1"
create_db_if_not_exists "praxia_shard2"

echo "All databases initialized successfully!"
