#!/usr/bin/env python3
"""
Configuration validation script for LLM Query Retrieval System.
This script validates the environment configuration and provides detailed feedback.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import validate_environment, get_settings
import json


def main():
    """
    Main function to validate configuration and provide detailed output.
    """
    print("🔍 LLM Query Retrieval System - Configuration Validation")
    print("=" * 60)
    
    try:
        # Validate environment
        result = validate_environment()
        
        print(f"✅ Configuration Status: {result['status'].upper()}")
        print()
        
        # Display configuration summary
        if result.get('config_summary'):
            print("📋 Configuration Summary:")
            print("-" * 30)
            for key, value in result['config_summary'].items():
                print(f"  {key}: {value}")
            print()
        
        # Display warnings if any
        if result.get('warnings'):
            print("⚠️  Warnings:")
            print("-" * 15)
            for warning in result['warnings']:
                print(f"  - {warning}")
            print()
        
        # Display errors if any
        if result.get('errors'):
            print("❌ Errors:")
            print("-" * 12)
            for error in result['errors']:
                print(f"  - {error}")
            print()
            return 1
        
        # Additional validation checks
        print("🔧 Additional Checks:")
        print("-" * 25)
        
        # Check if .env file exists
        env_file = project_root / ".env"
        if env_file.exists():
            print("  ✅ .env file found")
        else:
            print("  ⚠️  .env file not found (using system environment variables)")
        
        # Check required directories
        logs_dir = project_root / "logs"
        if not logs_dir.exists():
            print("  ℹ️  Creating logs directory...")
            logs_dir.mkdir(exist_ok=True)
            print("  ✅ Logs directory created")
        else:
            print("  ✅ Logs directory exists")
        
        # Test settings instantiation
        try:
            settings = get_settings()
            print("  ✅ Settings loaded successfully")
            
            # Environment-specific checks
            if settings.environment == "production":
                print("  🏭 Production environment detected")
                if settings.debug:
                    print("  ⚠️  Debug mode enabled in production (not recommended)")
                if "*" in settings.security.allowed_hosts:
                    print("  ⚠️  Wildcard allowed hosts in production (security risk)")
                if "*" in settings.security.cors_origins:
                    print("  ⚠️  Wildcard CORS origins in production (security risk)")
            elif settings.environment == "development":
                print("  🔧 Development environment detected")
                print("  ℹ️  Relaxed security settings are acceptable for development")
            
        except Exception as e:
            print(f"  ❌ Failed to load settings: {e}")
            return 1
        
        print()
        print("🎉 Configuration validation completed successfully!")
        
        # Provide next steps
        print()
        print("📋 Next Steps:")
        print("-" * 15)
        print("  1. Start the application:")
        print("     - Development: ./scripts/deploy.sh development")
        print("     - Production: ./scripts/deploy.sh production")
        print("  2. Test the health endpoint: curl http://localhost:8000/health")
        print("  3. View API documentation: http://localhost:8000/docs")
        
        return 0
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        print()
        print("🔧 Troubleshooting:")
        print("-" * 20)
        print("  1. Check that all required environment variables are set")
        print("  2. Verify API keys are valid and not empty")
        print("  3. Ensure database URL format is correct")
        print("  4. Copy .env.example to .env and configure it")
        print()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)