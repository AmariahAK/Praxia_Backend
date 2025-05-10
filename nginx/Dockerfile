FROM nginx:1.23-alpine

# Install dependencies
RUN apk add --no-cache openssl

# Remove default configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom configuration
COPY nginx.conf /etc/nginx/conf.d/

# Create directories for SSL certificates
RUN mkdir -p /etc/letsencrypt/live/praxia.example.com/
RUN mkdir -p /var/www/certbot

# Create self-signed certificates for development (will be replaced in production)
RUN openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/letsencrypt/live/praxia.example.com/privkey.pem \
    -out /etc/letsencrypt/live/praxia.example.com/fullchain.pem \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=praxia.example.com"

# Create custom error pages
RUN echo "<html><body><h1>404 - Page Not Found</h1><p>The page you requested could not be found.</p></body></html>" > /usr/share/nginx/html/404.html
RUN echo "<html><body><h1>500 - Server Error</h1><p>The server encountered an error and could not complete your request.</p></body></html>" > /usr/share/nginx/html/50x.html

# Expose ports
EXPOSE 80 443

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
