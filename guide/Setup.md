# Praxia Backend Setup Guide

This guide provides detailed instructions for setting up the Praxia Backend system in both development and production environments.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
- [Development Setup](#development-setup)
- [Production Setup](#production-setup)
- [Database Configuration](#database-configuration)
- [AI Model Configuration](#ai-model-configuration)
- [Monitoring Setup](#monitoring-setup)
- [Troubleshooting](#troubleshooting)
- [Dynamic Environment Configuration](#dynamic-environment-configuration)

## Prerequisites

Before setting up Praxia Backend, ensure you have the following installed:

- Docker and Docker Compose
- Git
- Python 3.9+ (for local development without Docker)
- PostgreSQL (for production setup)
- Together AI API key (for AI functionality)

## Environment Variables

Praxia uses environment variables for configuration. Create either a `.env` (development) or `.env.prod` (production) file in the project root with the following variables:

### Example `.env` for Development

```
# Django settings
SECRET_KEY=your-development-secret-key
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1,web

# Database settings
DB_ENGINE=django.db.backends.postgresql
DB_NAME=praxia_db
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=db
DB_PORT=5432

# Database connection pooling
DB_CONN_MAX_AGE=600
DB_CONNECT_TIMEOUT=10
DB_KEEPALIVES=1
DB_KEEPALIVES_IDLE=30
DB_KEEPALIVES_INTERVAL=10
DB_KEEPALIVES_COUNT=5
DB_ATOMIC_REQUESTS=False
DB_AUTOCOMMIT=True

# Sharding settings
USE_SHARDING=False
SHARD1_DB_NAME=praxia_shard1
SHARD1_DB_USER=your_db_user
SHARD1_DB_PASSWORD=your_db_password
SHARD1_DB_HOST=db
SHARD1_DB_PORT=5432
SHARD2_DB_NAME=praxia_shard2
SHARD2_DB_USER=your_db_user
SHARD2_DB_PASSWORD=your_db_password
SHARD2_DB_HOST=db
SHARD2_DB_PORT=5432

# Redis and Celery
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/1
CELERY_RESULT_BACKEND=redis://redis:6379/2
REDIS_HOST=redis
REDIS_PORT=6379

# CORS settings
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
CORS_ALLOW_ALL_ORIGINS=True

# AI API Keys
TOGETHER_AI_API_KEY=your_together_ai_api_key
TOGETHER_AI_MODEL=Qwen/Qwen3-235B-A22B-fp8-tput
INITIALIZE_XRAY_MODEL=True

# Admin user
DJANGO_SUPERUSER_USERNAME=admin
DJANGO_SUPERUSER_EMAIL=admin@example.com
DJANGO_SUPERUSER_PASSWORD=secure_password

# File upload limits
DATA_UPLOAD_MAX_MEMORY_SIZE=10485760
FILE_UPLOAD_MAX_MEMORY_SIZE=10485760

# Rate limiting
REST_FRAMEWORK_DEFAULT_THROTTLE_RATES_ANON=3/minute
REST_FRAMEWORK_DEFAULT_THROTTLE_RATES_USER=60/minute
REST_FRAMEWORK_DEFAULT_THROTTLE_RATES_AI_CONSULTATION=10/minute
REST_FRAMEWORK_DEFAULT_THROTTLE_RATES_AI_XRAY=5/hour
REST_FRAMEWORK_DEFAULT_THROTTLE_RATES_AI_RESEARCH=20/hour
REST_FRAMEWORK_DEFAULT_THROTTLE_RATES_AI_CHAT=30/minute

# Health check
HEALTH_CHECK_INTERVAL=840

# CDN settings
USE_CDN=False
CDN_URL=https://cdn.praxia.example.com

# AWS S3 settings
USE_S3=False
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_STORAGE_BUCKET_NAME=praxia-static
AWS_S3_CUSTOM_DOMAIN=cdn.praxia.example.com
AWS_S3_OBJECT_CACHE_CONTROL=max-age=86400

# Monitoring
PROMETHEUS_EXPORT_METRICS=True

# LibreTranslate settings
LIBRETRANSLATE_URL=http://libretranslate:5000

# Email Configs
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=smtp.example.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=your_email@example.com
EMAIL_HOST_PASSWORD=your_email_password
DEFAULT_FROM_EMAIL=Praxia <noreply@example.com>

# 2FA
OTP_TOTP_ISSUER=Praxia Health

# Health news settings
HEALTH_NEWS_SOURCES=who,cdc
HEALTH_NEWS_CACHE_TIMEOUT=43200

# Language and timezone
LANGUAGE_CODE=en-us
TIME_ZONE=UTC
USE_I18N=True
USE_TZ=True

# Celery settings
CELERY_ACCEPT_CONTENT=json
CELERY_TASK_SERIALIZER=json
CELERY_RESULT_SERIALIZER=json
```

### Example `.env.prod` for Production

For production, create a `.env.prod` file with similar variables but with these important differences:

```
# Django settings
SECRET_KEY=your-secure-production-secret-key
DEBUG=False
ALLOWED_HOSTS=your-domain.com,www.your-domain.com

# External Database settings (must be pre-configured)
DB_ENGINE=django.db.backends.postgresql
DB_NAME=praxia_db
DB_USER=your_prod_db_user
DB_PASSWORD=your_secure_prod_db_password
DB_HOST=your-production-db-host
DB_PORT=5432

# Sharding settings (must be pre-configured)
USE_SHARDING=True
SHARD1_DB_NAME=praxia_shard1
SHARD1_DB_USER=your_prod_db_user
SHARD1_DB_PASSWORD=your_secure_prod_db_password
SHARD1_DB_HOST=your-production-db-host
SHARD1_DB_PORT=5432
SHARD2_DB_NAME=praxia_shard2
SHARD2_DB_USER=your_prod_db_user
SHARD2_DB_PASSWORD=your_secure_prod_db_password
SHARD2_DB_HOST=your-production-db-host
SHARD2_DB_PORT=5432

# Other settings remain similar but with production-appropriate values
```

## Development Setup

In development mode, the system will automatically create the necessary databases in the PostgreSQL container.

### Initial Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/AmariahAK/Praxia_Backend.git
   cd Praxia_Backend
   ```

2. Create a `.env` file in the project root using the example above.

3. Make sure all script files are executable:
   ```bash
   chmod +x docker-entrypoint-wrapper.sh
   chmod +x entrypoint.sh
   chmod +x entrypoint.prod.sh
   chmod +x init-db.sh
   ```

4. Create necessary directories:
   ```bash
   mkdir -p media/profile_pics
   mkdir -p media/xrays
   mkdir -p data/models
   mkdir -p staticfiles
   mkdir -p prometheus
   ```

5. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

6. The API will be available at `http://localhost:8000/api/`

7. Access the admin interface at `http://localhost:8000/admin/` using the superuser credentials defined in your `.env` file.

8. Monitoring dashboards:
   - Prometheus: `http://localhost:9090/`
   - Grafana: `http://localhost:3000/` (default login: admin / admin_password)

### Local Development Without Docker

For local development without Docker:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up a local PostgreSQL database and update your `.env` file accordingly.

4. Apply migrations:
   ```bash
   python manage.py migrate
   ```

5. Create a superuser:
   ```bash
   python manage.py createsuperuser
   ```

6. Run the development server:
   ```bash
   python manage.py runserver
   ```

## Production Setup

Production setup requires pre-configured external PostgreSQL databases (main database and shards if using sharding).

1. Clone the repository:
   ```bash
   git clone https://github.com/AmariahAK/Praxia_Backend.git
   cd Praxia_Backend
   ```

2. Create a `.env.prod` file in the project root using the production example above.

3. Make sure all script files are executable:
   ```bash
   chmod +x docker-entrypoint-wrapper.sh
   chmod +x entrypoint.sh
   chmod +x entrypoint.prod.sh
   chmod +x init-db.sh
   ```

4. Configure your external PostgreSQL databases:
   - Create the main database (praxia_db by default)
   - If using sharding, create the shard databases (praxia_shard1 and praxia_shard2 by default)
   - Ensure the database user has appropriate permissions

5. Build and start the production containers:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

6. Set up SSL certificates:
   - The production setup includes Certbot for Let's Encrypt certificates
   - Update the domain in `nginx/nginx.conf` to match your actual domain
   - Initial certificates will be self-signed; they'll be replaced with Let's Encrypt certificates

7. The API will be available at `https://your-domain.com/api/`

8. Access the admin interface at `https://your-domain.com/admin/` using the superuser credentials defined in your `.env.prod` file.

## Dynamic Environment Configuration

Praxia Backend uses a dynamic environment configuration system that automatically selects the appropriate entrypoint script and environment file based on which docker-compose file is used.

### How It Works

1. The system uses a wrapper script (`docker-entrypoint-wrapper.sh`) that determines which environment is being used.
2. For development (using `docker-compose.yml`):
   - Uses `entrypoint.sh` as the entrypoint script
   - Uses `.env` for environment variables
3. For production (using `docker-compose.prod.yml`):
   - Uses `entrypoint.prod.sh` as the entrypoint script
   - Uses `.env.prod` for environment variables

### Usage

- For development:
  ```bash
  docker-compose up --build
  ```

- For production:
  ```bash
  docker-compose -f docker-compose.prod.yml up --build
  ```

The system automatically detects which environment is being used based on the `ENVIRONMENT` variable set in the docker-compose files.

## Database Configuration

### Development Database

In development mode:
- PostgreSQL runs in a Docker container
- Databases are automatically created if they don't exist
- Default credentials are specified in your `.env` file
- Database data is persisted in a Docker volume

### Production Database

For production:
- You must set up external PostgreSQL databases before deployment
- The system expects the databases to already exist
- Connection details are specified in your `.env.prod` file
- Consider implementing database backups and replication

### Database Sharding

Praxia supports database sharding for improved performance with large datasets:
- Enable sharding by setting `USE_SHARDING=True` in your environment file
- Configure the shard database connections (`SHARD1_*` and `SHARD2_*` variables)
- The system will automatically distribute data across shards based on user IDs

## AI Model Configuration

Praxia uses several AI models for different functions:

### Together AI Integration
- Sign up for a Together AI account at [together.ai](https://together.ai)
- Generate an API key
- Add your API key to the `.env` file as `TOGETHER_AI_API_KEY`
- Configure the model with `TOGETHER_AI_MODEL` (default: `Qwen/Qwen3-235B-A22B-fp8-tput`)

### X-ray Analysis Model
The system uses DenseNet121 for X-ray image analysis:
- Model weights are automatically downloaded during container startup
- To disable X-ray model initialization, set `INITIALIZE_XRAY_MODEL=False`
- Model weights are stored in `/app/data/models/`

## Monitoring Setup

Praxia includes comprehensive monitoring with Prometheus and Grafana:
- Prometheus metrics are available at `http://localhost:9090/` (development) or through your domain in production
- Grafana dashboards are available at `http://localhost:3000/` (development) or through your domain in production
- Default Grafana login is `admin` / `admin_password` (configurable via `GRAFANA_ADMIN_PASSWORD`)

### Available Metrics
- System metrics (CPU, memory, disk usage)
- Container metrics
- PostgreSQL metrics
- Redis metrics
- Django application metrics
- API endpoint performance
- AI model performance

## Troubleshooting

### Common Issues

#### Permission Denied for Entrypoint Scripts
If you encounter `exec /app/docker-entrypoint-wrapper.sh: permission denied` errors:

```bash
# Make all script files executable
chmod +x docker-entrypoint-wrapper.sh
chmod +x entrypoint.sh
chmod +x entrypoint.prod.sh
chmod +x init-db.sh
```

#### Database Does Not Exist
If you see errors like `database "praxia_db" does not exist`:

1. Make sure your init-db.sh script is executable:
   ```bash
   chmod +x init-db.sh
   ```

2. Verify the database settings in your .env file match what's in init-db.sh

3. You can manually create the database:
   ```bash
   docker-compose exec db psql -U your_db_user -c "CREATE DATABASE praxia_db;"
   ```

#### MONAI Container Syntax Error
If you see a syntax error in the MONAI container:

Update the command in docker-compose.yml to use proper syntax:
```yml
command: >
  python -c "import monai; print('MONAI initialized successfully'); import time; while True: time.sleep(3600)"
```

#### Missing Directories
If containers fail because of missing directories:

```bash
# Create necessary directories
mkdir -p media/profile_pics
mkdir -p media/xrays
mkdir -p data/models
mkdir -p staticfiles
mkdir -p prometheus
```

#### Database Connection Errors
If you encounter database connection errors:
- Verify database credentials in your `.env` or `.env.prod` file
- Check that the database server is running and accessible
- For production, ensure the database user has appropriate permissions
- Check network connectivity between containers or to external database

#### AI Model Errors
If AI functionality is not working:
- Verify your Together AI API key is correct
- Check that the specified model exists and is available
- Look for error messages in the logs with `docker-compose logs web`
- Ensure the model weights were downloaded successfully

#### Container Startup Issues
If containers fail to start:
- Check logs with `docker-compose logs`
- Verify all required environment variables are set
- Ensure ports are not already in use by other services
- Check disk space and system resources

#### Volume Mount Issues
If you're experiencing issues with volume mounts:

1. Use the `:delegated` option for better performance:
   ```yml
   volumes:
     - ./:/app:delegated
   ```

2. Ensure the host directories exist before mounting:
   ```bash
   mkdir -p media staticfiles data/models prometheus
   ```

3. Check for permission issues on the host directories:
   ```bash
   chmod -R 755 media staticfiles data prometheus
   ```

#### Docker Image Download Timeouts
If you experience timeouts when downloading Docker images (especially with slower internet connections):

**Recommended approach**: Pull large images separately before starting the full stack:

```bash
# Pull the largest images individually first
docker pull projectmonai/monai:latest
docker pull libretranslate/libretranslate
docker pull grafana/grafana
docker pull prom/prometheus
```

Then start the full stack:

```bash
docker compose up --build
```

Alternatively, you can start services incrementally:

```bash
# Start core services first
docker compose up -d db redis
# Then add application services
docker compose up -d web celery celery-beat
# Finally add monitoring and AI services
docker compose up -d monai prometheus grafana libretranslate
```

### Automated Setup Script

For convenience, you can create a setup script to automate the initial configuration:

```bash
#!/bin/bash

MODE=$1

if [ "$MODE" = "dev" ]; then
    echo "Setting up development environment..."
    
    # Make sure all scripts are executable
    chmod +x docker-entrypoint-wrapper.sh
    chmod +x entrypoint.sh
    chmod +x entrypoint.prod.sh
    chmod +x init-db.sh
    
    # Create necessary directories
    mkdir -p media/profile_pics
    mkdir -p media/xrays
    mkdir -p data/models
    mkdir -p staticfiles
    mkdir -p prometheus
    
    # Create prometheus config if it doesn't exist
    if [ ! -f prometheus/prometheus.yml ]; then
        cp prometheus/prometheus.yml.example prometheus/prometheus.yml || echo "prometheus.yml.example not found"
    fi
    
    echo "Development environment setup complete!"
    echo "Run 'docker-compose up' to start the development environment."
    
elif [ "$MODE" = "prod" ]; then
    echo "Setting up production environment..."
    
    # Make sure all scripts are executable
    chmod +x docker-entrypoint-wrapper.sh
    chmod +x entrypoint.sh
    chmod +x entrypoint.prod.sh
    chmod +x init-db.sh
    
    echo "Production environment setup complete!"
    echo "Run 'docker-compose -f docker-compose.prod.yml up -d' to start the production environment."
    
else
    echo "Usage: ./setup-env.sh [dev|prod]"
    exit 1
fi
```

Save this as `setup-env.sh` and make it executable:

```bash
chmod +x setup-env.sh
```

Then run it before starting your environment:

```bash
./setup-env.sh dev  # For development
./setup-env.sh prod  # For production
```

### Logs
Access logs for troubleshooting:
```bash
# View logs for all containers
docker-compose logs

# View logs for a specific container
docker-compose logs web

# Follow logs in real-time
docker-compose logs -f web
```

### Health Check
The system includes a health check endpoint at `/api/health/` that provides status information for all components.

## Recommended Docker Compose Changes

For more reliable development setup, consider these changes to your docker-compose.yml:

### Web Service
```yml
web:
  build: .
  volumes:
    - ./:/app:delegated  # Use delegated for better performance
    - static_volume:/app/staticfiles
    - media_volume:/app/media
  ports:
    - "8000:8000"
  env_file:
    - ./.env
  environment:
    - ENVIRONMENT=development
  depends_on:
    db:
      condition: service_healthy
    redis:
      condition: service_started
  restart: always
  networks:
    - praxia-network
```

### Database Service
```yml
db:
  image: postgres:14
  volumes:
    - postgres_data:/var/lib/postgresql/data/
    - ./init-db.sh:/docker-entrypoint-initdb.d/init-db.sh
  env_file:
    - ./.env
  environment:
    - POSTGRES_PASSWORD=${DB_PASSWORD}
    - POSTGRES_USER=${DB_USER}
    - POSTGRES_DB=${DB_NAME}
  ports:
    - "5432:5432"
  restart: always
  networks:
    - praxia-network
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
    interval: 5s
    timeout: 5s
    retries: 5
```

### MONAI Service
```yml
monai:
  image: projectmonai/monai:latest
  platform: linux/amd64
  ports:
    - "8888:8888"
  volumes:
    - ./data/models:/models
  networks:
    - praxia-network
  shm_size: "1g"
  ipc: host
  ulimits:
    memlock: -1
    stack: 67108864
  command: >
    python -c "import monai; print('MONAI initialized successfully'); import time; while True: time.sleep(3600)"
```

## Development vs Production Differences

### Development Mode
- Uses local Docker containers for all services
- Creates databases automatically
- Stores data in Docker volumes
- Uses `.env` for configuration
- Mounts local directories into containers for live code changes
- Debug mode enabled

### Production Mode
- Uses external databases (pre-configured)
- Uses Nginx for SSL termination and serving static files
- Uses Certbot for SSL certificates
- Uses `.env.prod` for configuration
- No code mounting (uses built Docker images)
- Debug mode disabled
- Implements proper security measures

By following this guide and using the troubleshooting tips, you should be able to successfully set up and run the Praxia Backend in both development and production environments.

---

[← Back to Main README](../README.md)
