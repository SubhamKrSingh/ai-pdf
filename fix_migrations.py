#!/usr/bin/env python3
"""
Script to fix migration state when tables already exist.
This marks existing tables as migrated in the migration tracking system.
"""

import asyncio
import asyncpg
from app.config import get_settings

async def fix_migration_state():
    """Mark existing migrations as applied."""
    settings = get_settings()
    conn = await asyncpg.connect(settings.database_url)
    
    try:
        # Create migrations table if it doesn't exist
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id SERIAL PRIMARY KEY,
                migration_name VARCHAR(255) NOT NULL UNIQUE,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checksum VARCHAR(64)
            )
        """)
        
        # Check which tables exist
        existing_tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
        """)
        
        table_names = [row['table_name'] for row in existing_tables]
        print(f"Existing tables: {table_names}")
        
        # Mark migrations as applied based on existing tables
        migrations_to_mark = []
        
        if 'documents' in table_names:
            migrations_to_mark.append("001_create_documents_table")
            
        if 'query_sessions' in table_names:
            migrations_to_mark.append("002_create_query_sessions_table")
            
        if 'document_url_cache' in table_names:
            migrations_to_mark.append("003_create_document_url_cache_table")
        
        # Always mark extensions as applied (they're idempotent)
        migrations_to_mark.append("000_create_extensions")
        
        # Insert migration records
        for migration_name in migrations_to_mark:
            try:
                await conn.execute("""
                    INSERT INTO schema_migrations (migration_name, applied_at)
                    VALUES ($1, CURRENT_TIMESTAMP)
                    ON CONFLICT (migration_name) DO NOTHING
                """, migration_name)
                print(f"Marked migration as applied: {migration_name}")
            except Exception as e:
                print(f"Error marking migration {migration_name}: {e}")
        
        # Show current migration status
        rows = await conn.fetch("""
            SELECT migration_name, applied_at
            FROM schema_migrations
            ORDER BY applied_at
        """)
        
        print("\nCurrent migration status:")
        for row in rows:
            print(f"  {row['migration_name']} - {row['applied_at']}")
            
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(fix_migration_state())