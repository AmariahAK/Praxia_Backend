#!/bin/bash

# Make script executable
chmod +x setup-env.sh

MODE=$1

if [ "$MODE" = "dev" ]; then
    echo "Setting up development environment..."
    
    # Make sure all scripts are executable
    chmod +x docker-entrypoint-wrapper.sh
    chmod +x entrypoint.sh
    chmod +x entrypoint.prod.sh
    
    # Create necessary directories
    mkdir -p media/profile_pics
    mkdir -p media/xrays
    mkdir -p data/models
    mkdir -p staticfiles
    mkdir -p prometheus
    
    # Create prometheus config if it doesn't exist
    if [ ! -f prometheus/prometheus.yml ]; then
        cp prometheus/prometheus.yml.example prometheus/prometheus.yml
    fi
    
    echo "Development environment setup complete!"
    echo "Run 'docker-compose up' to start the development environment."
    
elif [ "$MODE" = "prod" ]; then
    echo "Setting up production environment..."
    
    # Make sure all scripts are executable
    chmod +x docker-entrypoint-wrapper.sh
    chmod +x entrypoint.sh
    chmod +x entrypoint.prod.sh
    
    echo "Production environment setup complete!"
    echo "Run 'docker-compose -f docker-compose.prod.yml up -d' to start the production environment."
    
else
    echo "Usage: ./setup-env.sh [dev|prod]"
    exit 1
fi
