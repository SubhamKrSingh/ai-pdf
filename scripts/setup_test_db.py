#!/usr/bin/env python3
"""Setup test database for testing."""

import asyncio
import asyncpg
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.data.migrations import create_tables

# Test database configuration
TEST_DB_CONFIG = {
    "host": os.getenv("TEST_DB_HOST", "localhost"),
    "port": int(os.getenv("TEST_DB_PORT", "5432")),
    "user": os.getenv("TEST_DB_USER", "postgres"),
    "password": os.getenv("TEST_DB_PASSWORD", "postgres"),
    "database": "postgres",  # Connect to postgres to create test db
}

TEST_DB_NAME = os.getenv("TEST_DB_NAME", "test_llm_system")

async def setup_test_database():
    """Set up the test database."""
    try:
        # Connect to PostgreSQL server
        conn = await asyncpg.connect(**TEST_DB_CONFIG)
        
        # Drop test database if it exists
        await conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
        print(f"Dropped existing test database: {TEST_DB_NAME}")
        
        # Create test database
        await conn.execute(f"CREATE DATABASE {TEST_DB_NAME}")
        print(f"Created test database: {TEST_DB_NAME}")
        
        await conn.close()
        
        # Connect to the test database and create tables
        test_config = TEST_DB_CONFIG.copy()
        test_config["database"] = TEST_DB_NAME
        
        test_conn = await asyncpg.connect(**test_config)
        
        # Create tables using the migration script
        await create_tables(test_conn)
        print("Created database tables")
        
        await test_conn.close()
        
        print(f"✅ Test database setup completed successfully!")
        print(f"Database: {TEST_DB_NAME}")
        print(f"Host: {TEST_DB_CONFIG['host']}:{TEST_DB_CONFIG['port']}")
        
        # Print environment variables for testing
        print("\nSet these environment variables for testing:")
        print(f"export TEST_DB_HOST={TEST_DB_CONFIG['host']}")
        print(f"export TEST_DB_PORT={TEST_DB_CONFIG['port']}")
        print(f"export TEST_DB_USER={TEST_DB_CONFIG['user']}")
        print(f"export TEST_DB_PASSWORD={TEST_DB_CONFIG['password']}")
        print(f"export TEST_DB_NAME={TEST_DB_NAME}")
        
    except Exception as e:
        print(f"❌ Failed to setup test database: {e}")
        print("\nMake sure PostgreSQL is running and accessible with the provided credentials.")
        print("You can also run tests without a real database - they will use mocks.")
        return False
    
    return True

async def cleanup_test_database():
    """Clean up the test database."""
    try:
        conn = await asyncpg.connect(**TEST_DB_CONFIG)
        await conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
        await conn.close()
        print(f"✅ Test database {TEST_DB_NAME} cleaned up successfully!")
    except Exception as e:
        print(f"❌ Failed to cleanup test database: {e}")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        asyncio.run(cleanup_test_database())
    else:
        asyncio.run(setup_test_database())

if __name__ == "__main__":
    main()