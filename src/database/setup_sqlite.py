#!/usr/bin/env python3
"""
SQLite Database Setup Script for LLM Hub
This script helps you set up and manage the SQLite database for the LLM Hub project.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import database_setup
sys.path.append(str(Path(__file__).parent))

from database_setup import DatabaseManager, create_tables, drop_tables

def main():
    """Main function to handle database setup"""
    print("=== LLM Hub SQLite Database Setup ===")
    
    # Check if database file already exists
    db_file = "llm_hub.db"
    if os.path.exists(db_file):
        print(f"Database file '{db_file}' already exists.")
        choice = input("Do you want to:\n1. Recreate the database (this will delete all data)\n2. Keep existing database\nEnter your choice (1 or 2): ")
        
        if choice == "1":
            print("Dropping existing tables...")
            drop_tables()
            print("Creating new tables...")
            create_tables()
        else:
            print("Keeping existing database. No changes made.")
    else:
        print(f"Creating new SQLite database: {db_file}")
        create_tables()
    
    print("\n=== Database Setup Complete ===")
    print(f"Database file: {os.path.abspath(db_file)}")
    print("You can now use the database with your LLM Hub application!")

def show_database_info():
    """Show information about the current database"""
    db_file = "llm_hub.db"
    if os.path.exists(db_file):
        size = os.path.getsize(db_file)
        print(f"Database file: {db_file}")
        print(f"Size: {size} bytes ({size/1024:.2f} KB)")
        print(f"Location: {os.path.abspath(db_file)}")
    else:
        print("No database file found. Run setup first.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "info":
            show_database_info()
        elif sys.argv[1] == "drop":
            print("Dropping all tables...")
            drop_tables()
        else:
            print("Usage: python setup_sqlite.py [info|drop]")
            print("  info - Show database information")
            print("  drop - Drop all tables")
    else:
        main() 