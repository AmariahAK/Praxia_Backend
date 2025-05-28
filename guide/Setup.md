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

## Prerequisites

Before setting up Praxia Backend, ensure you have the following installed:

- Docker and Docker Compose
- Git
- Python 3.9+ (for local development without Docker)
- PostgreSQL (for production setup)
- Together AI API key (for AI functionality)

## Environment Variables

Praxia uses environment variables for configuration. Create either a `.env` (development) or `.env.prod` (production) file in the project root.

### Example `.env` for Development

```env
# Django settings
SECRET_KEY=your-development-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1,web

# External Database settings
DB_ENGINE=django.db.backends.postgresql
DB_NAME=praxia_backend
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=host.docker.internal
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
SHARD1_DB_HOST=host.docker.internal
SHARD1_DB_PORT=5432
SHARD2_DB_NAME=praxia_shard2
SHARD2_DB_USER=your_db_user
SHARD2_DB_PASSWORD=your_db_password
SHARD2_DB_HOST=host.docker.internal
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
TOGETHER_AI_API_KEY=your_together_ai_api_key_here
TOGETHER_AI_MODEL=deepseek-ai/DeepSeek-V3
INITIALIZE_XRAY_MODEL=True

# Admin user
DJANGO_SUPERUSER_USERNAME=admin
DJANGO_SUPERUSER_EMAIL=admin@example.com
DJANGO_SUPERUSER_PASSWORD=your_secure_admin_password

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
CDN_URL=https://cdn.example.com

# AWS S3 settings
USE_S3=False
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_STORAGE_BUCKET_NAME=your-s3-bucket
AWS_S3_CUSTOM_DOMAIN=cdn.example.com
AWS_S3_OBJECT_CACHE_CONTROL=max-age=86400

# Monitoring
PROMETHEUS_EXPORT_METRICS=True

# LibreTranslate settings
LIBRETRANSLATE_URL=http://localhost:5000

# Email Configs
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=your_email@gmail.com
EMAIL_HOST_PASSWORD=your_app_password
DEFAULT_FROM_EMAIL="Your App Name <noreply@example.com>"

# 2FA
OTP_TOTP_ISSUER="Your App Name"

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

# Health check
HEALTH_CHECK_INTERVAL=21600

# Frontend URL
FRONTEND_URL=http://localhost:3000
```

### Example `.env.prod` for Production

```env
# Django settings
SECRET_KEY=your-secure-production-secret-key
DEBUG=False
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com

# External Database settings
DB_ENGINE=django.db.backends.postgresql
DB_NAME=praxia_backend
DB_USER=your_prod_db_user
DB_PASSWORD=your_secure_prod_db_password
DB_HOST=your-production-db-host.com
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
USE_SHARDING=True
SHARD1_DB_NAME=praxia_shard1
SHARD1_DB_USER=your_prod_db_user
SHARD1_DB_PASSWORD=your_secure_prod_db_password
SHARD1_DB_HOST=your-production-db-host.com
SHARD1_DB_PORT=5432
SHARD2_DB_NAME=praxia_shard2
SHARD2_DB_USER=your_prod_db_user
SHARD2_DB_PASSWORD=your_secure_prod_db_password
SHARD2_DB_HOST=your-production-db-host.com
SHARD2_DB_PORT=5432

# Redis and Celery
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/1
CELERY_RESULT_BACKEND=redis://redis:6379/2
REDIS_HOST=redis
REDIS_PORT=6379

# CORS settings
CORS_ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
CORS_ALLOW_ALL_ORIGINS=False

# AI API Keys
TOGETHER_AI_API_KEY=your-production-together-ai-api-key
TOGETHER_AI_MODEL=deepseek-ai/DeepSeek-V3
INITIALIZE_XRAY_MODEL=True

# Admin user
DJANGO_SUPERUSER_USERNAME=admin
DJANGO_SUPERUSER_EMAIL=admin@yourdomain.com
DJANGO_SUPERUSER_PASSWORD=your-secure-admin-password

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
USE_CDN=True
CDN_URL=https://cdn.yourdomain.com

# AWS S3 settings
USE_S3=True
AWS_ACCESS_KEY_ID=your_production_aws_access_key
AWS_SECRET_ACCESS_KEY=your_production_aws_secret_key
AWS_STORAGE_BUCKET_NAME=your-production-s3-bucket
AWS_S3_CUSTOM_DOMAIN=cdn.yourdomain.com
AWS_S3_OBJECT_CACHE_CONTROL=max-age=86400

# Monitoring
PROMETHEUS_EXPORT_METRICS=True

# LibreTranslate settings
LIBRETRANSLATE_URL=http://libretranslate:5000

# Email Configs
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=your-production-email@yourdomain.com
EMAIL_HOST_PASSWORD=your-production-app-password
DEFAULT_FROM_EMAIL="Your App Name <noreply@yourdomain.com>"

# 2FA
OTP_TOTP_ISSUER="Your App Name"

# Health news settings
HEALTH_NEWS_SOURCES=who,cdc,nih,mayo
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

# LibreTranslate settings
LT_LOAD_ONLY=es,en,fr
LT_THREADS=8
LT_MEMORY=200M
LT_UPDATE_MODELS=false

# Health check
HEALTH_CHECK_INTERVAL=21600

# Frontend URL
FRONTEND_URL=https://yourdomain.com
```

## Development Setup

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
   ```

4. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

5. The API will be available at `http://localhost:8000/api/`

6. Access the admin interface at `http://localhost:8000/admin/` using the superuser credentials defined in your `.env` file.

7. Monitoring dashboards:
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

Production setup requires pre-configured external PostgreSQL databases.

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
   ```

4. Configure your external PostgreSQL databases:
   - Create the main database (praxia_backend by default)
   - If using sharding, create the shard databases (praxia_shard1 and praxia_shard2 by default)
   - Ensure the database user has appropriate permissions

5. Build and start the production containers:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

6. Set up SSL certificates:
   - The production setup includes Nginx for SSL termination
   - Update the domain in `nginx/nginx.conf` to match your actual domain
   - Configure SSL certificates as needed

7. The API will be available at `https://yourdomain.com/api/`

8. Access the admin interface at `https://yourdomain.com/admin/` using the superuser credentials defined in your `.env.prod` file.

## Database Configuration

### Development Database

In development mode:
- PostgreSQL can run in a Docker container or be external
- Database connection details are specified in your `.env` file
- Migrations are applied automatically during container startup

### Production Database

For production:
- You must set up external PostgreSQL databases before deployment
- The system expects the databases to already exist
- Connection details are specified in your `.env.prod` file
- Consider implementing database backups and replication

### Database Sharding

Praxia supports database sharding for improved performance:
- Enable sharding by setting `USE_SHARDING=True` in your environment file
- Configure the shard database connections (`SHARD1_*` and `SHARD2_*` variables)
- The system will automatically distribute data across shards based on user IDs

## AI Model Configuration

Praxia uses several AI models for different functions:

### Together AI Integration
- Sign up for a Together AI account at [together.ai](https://together.ai)
- Generate an API key
- Add your API key to the `.env` file as `TOGETHER_AI_API_KEY`
- Configure the model with `TOGETHER_AI_MODEL` (default: `deepseek-ai/DeepSeek-V3`)

### X-ray Analysis Model
The system uses DenseNet121 for X-ray image analysis:
- Model weights are automatically downloaded during container startup
- To disable X-ray model initialization, set `INITIALIZE_XRAY_MODEL=False`
- Model weights are stored in `data/models/`

### AI Identity Customization
Developers can customize the AI assistant's identity by editing `data/ai_identity.txt`. This file contains:
- AI assistant name and description
- Developer information
- Primary healthcare functions
- Personality traits and response guidelines

## Monitoring Setup

Praxia includes comprehensive monitoring with Prometheus and Grafana:
- Prometheus metrics are available at `http://localhost:9090/` (development)
- Grafana dashboards are available at `http://localhost:3000/` (development)
- Default Grafana login is `admin` / `admin_password`

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
If you encounter permission denied errors:

```bash
chmod +x docker-entrypoint-wrapper.sh
chmod +x entrypoint.sh
chmod +x entrypoint.prod.sh
```

#### DenseNet Model Issues
If you encounter issues with the X-ray analysis model:

1. Manually run the model download script:
   ```bash
   python api/utils/download_model.py
   ```

2. Restart the Docker containers:
   ```bash
   docker-compose restart
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

## Development vs Production Differences

### Development Mode
- Uses local Docker containers for all services (optional)
- Uses `.env` for configuration
- Debug mode enabled
- CORS allows all origins
- Simplified SSL setup

### Production Mode
- Uses external databases (required)
- Uses Nginx for SSL termination and serving static files
- Uses `.env.prod` for configuration
- Debug mode disabled
- Restricted CORS origins
- Implements proper security measures

By following this guide, you should be able to successfully set up and run the Praxia Backend in both development and production environments.

---

[← Back to Main README](../README.md)
