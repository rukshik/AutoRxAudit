"""
Export Database Schema and Data
Extracts current working database schema and data for clean deployment
"""
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from api/.env
api_dir = Path(__file__).parent.parent / 'api'
env_path = api_dir / '.env'
load_dotenv(env_path)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'autorxaudit'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres')
}

print("Database Configuration:")
print(f"  Host: {DB_CONFIG['host']}")
print(f"  Database: {DB_CONFIG['database']}")
print(f"  User: {DB_CONFIG['user']}")
print()

def export_schema():
    """Export database schema (structure only)"""
    print("=" * 80)
    print("EXPORTING SCHEMA")
    print("=" * 80)
    
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    cursor = conn.cursor()
    
    schema_sql = []
    
    # Get all tables
    cursor.execute("""
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = 'public' 
        ORDER BY tablename
    """)
    tables = [row['tablename'] for row in cursor.fetchall()]
    
    print(f"\nFound {len(tables)} tables:")
    for table in tables:
        print(f"  - {table}")
    
    schema_sql.append("-- AutoRxAudit Database Schema")
    schema_sql.append("-- Exported from working database")
    schema_sql.append("-- " + "=" * 76)
    schema_sql.append("")
    
    # Export each table structure
    for table in tables:
        print(f"\nExporting schema for: {table}")
        
        # Get CREATE TABLE statement
        cursor.execute(f"""
            SELECT 
                'CREATE TABLE IF NOT EXISTS {table} (' ||
                string_agg(
                    column_name || ' ' || data_type ||
                    CASE 
                        WHEN character_maximum_length IS NOT NULL 
                        THEN '(' || character_maximum_length || ')'
                        ELSE ''
                    END ||
                    CASE 
                        WHEN is_nullable = 'NO' THEN ' NOT NULL'
                        ELSE ''
                    END ||
                    CASE 
                        WHEN column_default IS NOT NULL 
                        THEN ' DEFAULT ' || column_default
                        ELSE ''
                    END,
                    ', '
                ORDER BY ordinal_position
                ) || ');' as create_stmt
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = '{table}'
        """)
        
        result = cursor.fetchone()
        if result:
            schema_sql.append(f"-- {table.upper()} TABLE")
            schema_sql.append(result['create_stmt'])
            schema_sql.append("")
        
        # Get indexes
        cursor.execute(f"""
            SELECT indexdef 
            FROM pg_indexes 
            WHERE schemaname = 'public' AND tablename = '{table}'
            AND indexname NOT LIKE '%_pkey'
        """)
        indexes = cursor.fetchall()
        
        if indexes:
            for idx in indexes:
                schema_sql.append(idx['indexdef'] + ';')
            schema_sql.append("")
    
    cursor.close()
    conn.close()
    
    # Write schema file
    schema_file = api_dir / 'database' / 'schema.sql'
    with open(schema_file, 'w') as f:
        f.write('\n'.join(schema_sql))
    
    print(f"\n✓ Schema exported to: {schema_file}")
    return str(schema_file)


def export_data():
    """Export database data (excluding audit_logs)"""
    print("\n" + "=" * 80)
    print("EXPORTING DATA")
    print("=" * 80)
    
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    cursor = conn.cursor()
    
    data_sql = []
    data_sql.append("-- AutoRxAudit Database Data")
    data_sql.append("-- Exported from working database (excluding audit_logs)")
    data_sql.append("-- " + "=" * 76)
    data_sql.append("")
    
    # Tables to export (exclude audit_logs and prescription_requests - transactional data)
    tables_to_export = [
        'users',
        'patients',
        'admissions',
        'diagnoses',
        'prescriptions',
        'transfers',
        'omr',
        'drgcodes'
    ]
    
    for table in tables_to_export:
        # Check if table exists
        cursor.execute(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = '{table}'
            ) as exists
        """)
        
        result = cursor.fetchone()
        if not result['exists']:
            print(f"  Skipping {table} (table not found)")
            continue
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
        count = cursor.fetchone()['count']
        
        print(f"\nExporting data from: {table} ({count:,} rows)")
        
        if count == 0:
            continue
        
        # Get column names
        cursor.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'public' AND table_name = '{table}'
            ORDER BY ordinal_position
        """)
        columns = [row['column_name'] for row in cursor.fetchall()]
        
        data_sql.append(f"-- {table.upper()} DATA ({count:,} rows)")
        data_sql.append(f"TRUNCATE TABLE {table} CASCADE;")
        
        # Fetch data in batches
        batch_size = 1000
        offset = 0
        
        while offset < count:
            cursor.execute(f"""
                SELECT * FROM {table}
                ORDER BY 1
                LIMIT {batch_size} OFFSET {offset}
            """)
            
            rows = cursor.fetchall()
            
            if rows:
                # Generate INSERT statements
                values_list = []
                for row in rows:
                    values = []
                    for col in columns:
                        val = row[col]
                        if val is None:
                            values.append('NULL')
                        elif isinstance(val, str):
                            # Escape single quotes
                            val_escaped = val.replace("'", "''")
                            values.append(f"'{val_escaped}'")
                        elif isinstance(val, bool):
                            values.append('TRUE' if val else 'FALSE')
                        else:
                            values.append(str(val))
                    values_list.append(f"({', '.join(values)})")
                
                # Create INSERT statement
                cols_str = ', '.join(columns)
                values_str = ',\n    '.join(values_list)
                data_sql.append(f"INSERT INTO {table} ({cols_str}) VALUES")
                data_sql.append(f"    {values_str};")
                data_sql.append("")
            
            offset += batch_size
            print(f"  Processed {min(offset, count):,} / {count:,} rows")
    
    cursor.close()
    conn.close()
    
    # Write data file
    data_file = api_dir / 'database' / 'data.sql'
    with open(data_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(data_sql))
    
    print(f"\n✓ Data exported to: {data_file}")
    return str(data_file)


def main():
    print("=" * 80)
    print("DATABASE EXPORT UTILITY")
    print("=" * 80)
    
    try:
        schema_file = export_schema()
        data_file = export_data()
        
        print("\n" + "=" * 80)
        print("EXPORT COMPLETE")
        print("=" * 80)
        print(f"\nSchema file: {schema_file}")
        print(f"Data file: {data_file}")
        print("\nNext steps:")
        print("1. Review the exported files")
        print("2. Move old database files to obsolete folder")
        print("3. Commit the clean schema and data files")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
