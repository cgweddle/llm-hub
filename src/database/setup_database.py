#!/usr/bin/env python3
"""
Database Setup Script for LLM Hub
Supports both SQLite (development) and PostgreSQL (production)
"""

import os
import sys
import argparse
from pathlib import Path

# Add the current directory to the path so we can import database_setup
sys.path.append(str(Path(__file__).parent))

from database_setup import DatabaseManager, create_tables, drop_tables, get_database_manager

def setup_development(args):
    """Set up SQLite database for development"""
    print("=== Setting up SQLite Database for Development ===")
    
    # Set environment variables for development
    os.environ['ENVIRONMENT'] = 'development'
    
    db_manager = get_database_manager(
        database_url=args.url, 
        environment='development'
    )
    info = db_manager.get_database_info()
    
    print(f"Database Type: {info['type']}")
    print(f"Database URL: {info['url']}")
    
    # Check if database file already exists
    db_file = "llm_hub.db"
    if os.path.exists(db_file) and not args.force:
        print(f"\nDatabase file '{db_file}' already exists.")
        choice = input("Do you want to:\n1. Recreate the database (this will delete all data)\n2. Keep existing database\nEnter your choice (1 or 2): ")
        
        if choice == "1":
            print("Dropping existing tables...")
            drop_tables(database_url=args.url, environment='development')
            print("Creating new tables...")
            create_tables(database_url=args.url, environment='development')
        else:
            print("Keeping existing database. No changes made.")
    else:
        if args.force and os.path.exists(db_file):
            print("Force flag set - recreating existing database...")
            drop_tables(database_url=args.url, environment='development')
        
        print(f"\nCreating new SQLite database: {db_file}")
        create_tables(database_url=args.url, environment='development')
    
    # Test connection
    if db_manager.test_connection():
        print("✓ Database connection successful!")
    else:
        print("✗ Database connection failed!")
    
    print("\n=== Development Database Setup Complete ===")
    print(f"Database file: {os.path.abspath(db_file)}")

def setup_production(args):
    """Set up PostgreSQL database for production"""
    print("=== Setting up PostgreSQL Database for Production ===")
    
    # Set environment variables for production
    os.environ['ENVIRONMENT'] = 'production'
    
    # Check if DATABASE_URL is set
    database_url = args.url or os.getenv('DATABASE_URL')
    if not database_url:
        print("ERROR: DATABASE_URL environment variable is not set!")
        print("Please set DATABASE_URL for your PostgreSQL connection.")
        print("Example: export DATABASE_URL='postgresql://user:password@localhost/llm_hub'")
        print("Or use: --url postgresql://user:password@localhost/llm_hub")
        return False
    
    db_manager = get_database_manager(
        database_url=database_url, 
        environment='production'
    )
    info = db_manager.get_database_info()
    
    print(f"Database Type: {info['type']}")
    print(f"Database URL: {info['url']}")
    
    # Test connection
    if not db_manager.test_connection():
        print("✗ Database connection failed!")
        print("Please check your DATABASE_URL and ensure PostgreSQL is running.")
        return False
    
    print("✓ Database connection successful!")
    
    # Create tables
    print("\nCreating tables...")
    create_tables(database_url=database_url, environment='production')
    
    print("\n=== Production Database Setup Complete ===")
    return True

def show_info(args):
    """Show database information"""
    print("=== Database Information ===")
    
    # Check environment
    environment = args.environment or os.getenv('ENVIRONMENT', 'development')
    print(f"Environment: {environment}")
    
    db_manager = get_database_manager(
        database_url=args.url, 
        environment=environment
    )
    info = db_manager.get_database_info()
    
    print(f"Database Type: {info['type']}")
    print(f"Database URL: {info['url']}")
    
    if environment == 'development':
        db_file = "llm_hub.db"
        if os.path.exists(db_file):
            size = os.path.getsize(db_file)
            print(f"Database file: {db_file}")
            print(f"Size: {size} bytes ({size/1024:.2f} KB)")
            print(f"Location: {os.path.abspath(db_file)}")
        else:
            print("No SQLite database file found.")
    
    # Test connection
    if db_manager.test_connection():
        print("✓ Database connection successful")
    else:
        print("✗ Database connection failed")

def drop_database(args):
    """Drop all tables from the database"""
    environment = args.environment or os.getenv('ENVIRONMENT', 'development')
    
    if not args.force:
        confirm = input(f"Are you sure you want to drop ALL tables from {environment} environment? This action cannot be undone! (yes/no): ")
        if confirm.lower() != 'yes':
            print("Operation cancelled.")
            return
    
    print(f"Dropping tables from {environment} environment...")
    drop_tables(database_url=args.url, environment=environment)
    print("✓ Tables dropped successfully!")

def parse_arguments():
    """Parse command line arguments using argparse"""
    parser = argparse.ArgumentParser(
        description="LLM Hub Database Setup Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_database.py                    # Set up SQLite database (development)
  python setup_database.py development        # Set up SQLite database
  python setup_database.py production         # Set up PostgreSQL database
  python setup_database.py info               # Show database information
  python setup_database.py drop               # Drop all tables (use with caution!)
  python setup_database.py --url sqlite:///custom.db  # Use custom database URL
  python setup_database.py --force            # Skip confirmation prompts
        """
    )
    
    parser.add_argument(
        'action',
        nargs='?',
        default='development',
        choices=['development', 'production', 'info', 'drop'],
        help='Action to perform (default: development)'
    )
    
    parser.add_argument(
        '--url',
        type=str,
        help='Custom database URL (overrides DATABASE_URL environment variable)'
    )
    
    parser.add_argument(
        '--environment',
        type=str,
        choices=['development', 'production'],
        help='Force environment type (overrides ENVIRONMENT variable)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force action without confirmation prompts'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()

def main():
    """Main function with argparse integration"""
    args = parse_arguments()
    
    try:
        if args.verbose:
            print(f"Action: {args.action}")
            if args.url:
                print(f"Custom URL: {args.url}")
            if args.environment:
                print(f"Environment: {args.environment}")
        
        # Handle different actions
        if args.action == 'development':
            setup_development(args)
        elif args.action == 'production':
            setup_production(args)
        elif args.action == 'info':
            show_info(args)
        elif args.action == 'drop':
            drop_database(args)
        else:
            print(f"Unknown action: {args.action}")
            return False
        
        return True
    
    except ValueError as e:
        print(f"Configuration error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 