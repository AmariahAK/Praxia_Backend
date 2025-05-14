FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV BASE_DIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat-traditional \
    postgresql-client \
    gcc \
    g++ \
    build-essential \
    libc6-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Make entrypoint scripts executable
RUN chmod +x /app/entrypoint.sh
RUN chmod +x /app/entrypoint.prod.sh

# Create necessary directories
RUN mkdir -p /app/media/profile_pics
RUN mkdir -p /app/media/xrays
RUN mkdir -p /app/data/models
RUN mkdir -p /app/staticfiles
RUN mkdir -p /app/prometheus

# Set permissions
RUN chmod -R 755 /app/media
RUN chmod -R 755 /app/staticfiles
RUN chmod -R 755 /app/data
RUN chmod -R 755 /app/prometheus

# Use the wrapper as the entrypoint
ENTRYPOINT ["/app/docker-entrypoint-wrapper.sh"]
