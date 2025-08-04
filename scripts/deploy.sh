#!/bin/bash

# Deployment script for LLM Query Retrieval System
# Usage: ./scripts/deploy.sh [environment]

set -e

ENVIRONMENT=${1:-production}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting deployment for environment: $ENVIRONMENT"

# Change to project directory
cd "$PROJECT_DIR"

# Validate environment file exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found. Please copy .env.example to .env and configure it."
    exit 1
fi

# Validate configuration
echo "🔍 Validating configuration..."
python -c "
from app.config import validate_environment
try:
    result = validate_environment()
    print('✅ Configuration validation passed')
    if result.get('warnings'):
        print('⚠️  Warnings:')
        for warning in result['warnings']:
            print(f'   - {warning}')
except Exception as e:
    print(f'❌ Configuration validation failed: {e}')
    exit(1)
"

# Build Docker images
echo "🏗️  Building Docker images..."
if [ "$ENVIRONMENT" = "development" ]; then
    docker-compose -f docker-compose.dev.yml build
else
    docker-compose build
fi

# Stop existing containers
echo "🛑 Stopping existing containers..."
if [ "$ENVIRONMENT" = "development" ]; then
    docker-compose -f docker-compose.dev.yml down
else
    docker-compose down
fi

# Start services
echo "🚀 Starting services..."
if [ "$ENVIRONMENT" = "development" ]; then
    docker-compose -f docker-compose.dev.yml up -d
else
    docker-compose up -d
fi

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check health
echo "🏥 Checking service health..."
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Service is healthy!"
        break
    fi
    
    if [ $attempt -eq $max_attempts ]; then
        echo "❌ Service failed to become healthy after $max_attempts attempts"
        echo "📋 Container logs:"
        if [ "$ENVIRONMENT" = "development" ]; then
            docker-compose -f docker-compose.dev.yml logs app
        else
            docker-compose logs app
        fi
        exit 1
    fi
    
    echo "⏳ Attempt $attempt/$max_attempts - waiting for service..."
    sleep 5
    ((attempt++))
done

# Run database migrations if needed
echo "🗄️  Running database migrations..."
if [ "$ENVIRONMENT" = "development" ]; then
    docker-compose -f docker-compose.dev.yml exec app python -c "
from app.data.migrations import run_migrations
run_migrations()
print('✅ Database migrations completed')
"
else
    docker-compose exec app python -c "
from app.data.migrations import run_migrations
run_migrations()
print('✅ Database migrations completed')
"
fi

echo "🎉 Deployment completed successfully!"
echo "📊 Service status:"
if [ "$ENVIRONMENT" = "development" ]; then
    docker-compose -f docker-compose.dev.yml ps
else
    docker-compose ps
fi

echo ""
echo "🌐 Service URLs:"
echo "   - API: http://localhost:8000"
echo "   - Health Check: http://localhost:8000/health"
echo "   - API Documentation: http://localhost:8000/docs"
echo ""
echo "📋 Useful commands:"
echo "   - View logs: docker-compose logs -f app"
echo "   - Stop services: docker-compose down"
echo "   - Restart services: docker-compose restart"