#!/bin/bash
set -e

# Create postgres role if it doesn't exist
echo "Creating postgres role if it doesn't exist..."
psql -v ON_ERROR_STOP=0 -U "$POSTGRES_USER" <<-EOSQL
    DO
    \$do\$
    BEGIN
       IF NOT EXISTS (
          SELECT FROM pg_catalog.pg_roles
          WHERE  rolname = 'postgres') THEN
          CREATE ROLE postgres WITH SUPERUSER LOGIN PASSWORD 'postgres_password';
       END IF;
    END
    \$do\$;
EOSQL

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

# Create the main database first
echo "Creating main database..."
psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE "$DB_NAME";
    GRANT ALL PRIVILEGES ON DATABASE "$DB_NAME" TO "$POSTGRES_USER";
    GRANT ALL PRIVILEGES ON DATABASE "$DB_NAME" TO postgres;
EOSQL

# Create user database if needed
echo "Creating user database if it doesn't exist..."
psql -v ON_ERROR_STOP=0 -U "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE "$POSTGRES_USER";
    GRANT ALL PRIVILEGES ON DATABASE "$POSTGRES_USER" TO postgres;
EOSQL

# Create shard databases if sharding is enabled
if [ "$USE_SHARDING" = "True" ]; then
    echo "Creating shard databases..."
    psql -v ON_ERROR_STOP=0 -U "$POSTGRES_USER" <<-EOSQL
        CREATE DATABASE "$SHARD1_DB_NAME";
        CREATE DATABASE "$SHARD2_DB_NAME";
        GRANT ALL PRIVILEGES ON DATABASE "$SHARD1_DB_NAME" TO "$POSTGRES_USER";
        GRANT ALL PRIVILEGES ON DATABASE "$SHARD2_DB_NAME" TO "$POSTGRES_USER";
        GRANT ALL PRIVILEGES ON DATABASE "$SHARD1_DB_NAME" TO postgres;
        GRANT ALL PRIVILEGES ON DATABASE "$SHARD2_DB_NAME" TO postgres;
EOSQL
fi

echo "All databases initialized successfully!"
