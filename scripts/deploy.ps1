# PowerShell deployment script for LLM Query Retrieval System
# Usage: .\scripts\deploy.ps1 [environment]

param(
    [string]$Environment = "production"
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 Starting deployment for environment: $Environment" -ForegroundColor Green

# Get script and project directories
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

# Change to project directory
Set-Location $ProjectDir

# Validate environment file exists
if (-not (Test-Path ".env")) {
    Write-Host "❌ Error: .env file not found. Please copy .env.example to .env and configure it." -ForegroundColor Red
    exit 1
}

# Validate configuration
Write-Host "🔍 Validating configuration..." -ForegroundColor Yellow
try {
    $validationResult = python -c @"
from app.config import validate_environment
try:
    result = validate_environment()
    print('Configuration validation passed')
    if result.get('warnings'):
        print('Warnings:')
        for warning in result['warnings']:
            print(f'   - {warning}')
except Exception as e:
    print(f'Configuration validation failed: {e}')
    exit(1)
"@
    Write-Host "✅ $validationResult" -ForegroundColor Green
} catch {
    Write-Host "❌ Configuration validation failed: $_" -ForegroundColor Red
    exit 1
}

# Build Docker images
Write-Host "🏗️  Building Docker images..." -ForegroundColor Yellow
if ($Environment -eq "development") {
    docker-compose -f docker-compose.dev.yml build
} else {
    docker-compose build
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed" -ForegroundColor Red
    exit 1
}

# Stop existing containers
Write-Host "🛑 Stopping existing containers..." -ForegroundColor Yellow
if ($Environment -eq "development") {
    docker-compose -f docker-compose.dev.yml down
} else {
    docker-compose down
}

# Start services
Write-Host "🚀 Starting services..." -ForegroundColor Green
if ($Environment -eq "development") {
    docker-compose -f docker-compose.dev.yml up -d
} else {
    docker-compose up -d
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to start services" -ForegroundColor Red
    exit 1
}

# Wait for services to be healthy
Write-Host "⏳ Waiting for services to be healthy..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check health
Write-Host "🏥 Checking service health..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 1

while ($attempt -le $maxAttempts) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ Service is healthy!" -ForegroundColor Green
            break
        }
    } catch {
        # Continue trying
    }
    
    if ($attempt -eq $maxAttempts) {
        Write-Host "❌ Service failed to become healthy after $maxAttempts attempts" -ForegroundColor Red
        Write-Host "📋 Container logs:" -ForegroundColor Yellow
        if ($Environment -eq "development") {
            docker-compose -f docker-compose.dev.yml logs app
        } else {
            docker-compose logs app
        }
        exit 1
    }
    
    Write-Host "⏳ Attempt $attempt/$maxAttempts - waiting for service..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    $attempt++
}

# Run database migrations if needed
Write-Host "🗄️  Running database migrations..." -ForegroundColor Yellow
try {
    if ($Environment -eq "development") {
        docker-compose -f docker-compose.dev.yml exec app python -c @"
from app.data.migrations import run_migrations
run_migrations()
print('Database migrations completed')
"@
    } else {
        docker-compose exec app python -c @"
from app.data.migrations import run_migrations
run_migrations()
print('Database migrations completed')
"@
    }
    Write-Host "✅ Database migrations completed" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Database migrations failed (this may be expected if already up to date): $_" -ForegroundColor Yellow
}

Write-Host "🎉 Deployment completed successfully!" -ForegroundColor Green
Write-Host "📊 Service status:" -ForegroundColor Cyan
if ($Environment -eq "development") {
    docker-compose -f docker-compose.dev.yml ps
} else {
    docker-compose ps
}

Write-Host ""
Write-Host "🌐 Service URLs:" -ForegroundColor Cyan
Write-Host "   - API: http://localhost:8000"
Write-Host "   - Health Check: http://localhost:8000/health"
Write-Host "   - API Documentation: http://localhost:8000/docs"
Write-Host ""
Write-Host "📋 Useful commands:" -ForegroundColor Cyan
Write-Host "   - View logs: docker-compose logs -f app"
Write-Host "   - Stop services: docker-compose down"
Write-Host "   - Restart services: docker-compose restart"