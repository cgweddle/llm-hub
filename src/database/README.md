# Database Setup for LLM Hub

This directory contains the database setup scripts for the LLM Hub project. The system supports both SQLite (for development) and PostgreSQL (for production) with enhanced security features.

## Quick Start

### For Development (SQLite)

```bash
# Install dependencies
pip install -r requirements.txt

# Set up SQLite database (default)
python setup_database.py
# or explicitly
python setup_database.py development
```

### For Production (PostgreSQL)

```bash
# Install dependencies
pip install -r requirements.txt

# Set your PostgreSQL connection string
export DATABASE_URL="postgresql://username:password@localhost/llm_hub"

# Set up PostgreSQL database
python setup_database.py production
```

## Available Scripts

### 1. `database_setup.py`
The core database module containing:
- SQLAlchemy models for all tables
- `DatabaseManager` class for database operations
- Support for both SQLite and PostgreSQL
- **Security features**: URL validation, input sanitization, parameterized queries

**Usage:**
```bash
# Create tables (defaults to development/SQLite)
python database_setup.py

# Create tables for specific environment
python database_setup.py development
python database_setup.py production

# Show database info
python database_setup.py info

# Drop all tables (with confirmation)
python database_setup.py drop

# Use custom database URL
python database_setup.py --url sqlite:///custom.db

# Force actions without prompts
python database_setup.py drop --force

# Enable verbose output
python database_setup.py --verbose
```

### 2. `setup_database.py`
A comprehensive setup script with interactive prompts and better error handling.

**Usage:**
```bash
# Set up development database (SQLite)
python setup_database.py development

# Set up production database (PostgreSQL)
python setup_database.py production

# Show database information
python setup_database.py info

# Drop all tables (with confirmation)
python setup_database.py drop

# Use custom database URL
python setup_database.py --url postgresql://user:pass@host/db

# Force actions without prompts
python setup_database.py --force
```

## Command Line Options

Both scripts support the following options:

- `--url`: Custom database URL (overrides environment detection)
- `--environment`: Force environment type (development/production)
- `--force`: Skip confirmation prompts
- `--verbose`: Enable verbose output and error details

## Environment Configuration

### Development Environment
- **Database**: SQLite
- **File**: `llm_hub.db` (created in current directory)
- **Environment Variable**: `ENVIRONMENT=development` (optional, default)

### Production Environment
- **Database**: PostgreSQL
- **Environment Variables**:
  - `ENVIRONMENT=production`
  - `DATABASE_URL=postgresql://user:password@host:port/database`

## Security Features

### Input Validation
- **URL Sanitization**: All database URLs are validated and sanitized
- **Path Traversal Protection**: SQLite paths are checked for directory traversal attempts
- **Hostname Validation**: PostgreSQL hostnames are validated for suspicious patterns
- **Parameterized Queries**: All SQL queries use parameterized statements to prevent injection

### Security Checks
- **SQLite Path Safety**: Prevents `..` in paths and absolute path access
- **PostgreSQL Hostname Filtering**: Blocks suspicious domains and special characters
- **Connection Testing**: Validates database connectivity before operations
- **Error Handling**: Comprehensive error handling without exposing sensitive information

### Best Practices
- **Environment Separation**: Clear separation between development and production
- **Confirmation Prompts**: Destructive operations require explicit confirmation
- **Graceful Degradation**: Proper error handling and exit codes
- **Logging Control**: SQL debugging can be enabled via `SQL_DEBUG=true`

## Database Schema

The database includes the following tables:

- **users**: User accounts and authentication
- **agents**: AI agents with configurations
- **tools**: Available tools and functions
- **flows**: Workflow definitions
- **executions**: Execution history and results
- **messages**: Conversation messages
- **agent_tool_association**: Many-to-many relationship between agents and tools
- **agent_flow_association**: Many-to-many relationship between agents and flows

## Using the Database in Your Application

```python
from database_setup import get_database_manager

# Get database manager (auto-detects environment)
db_manager = get_database_manager()

# Or specify environment
db_manager = get_database_manager(environment='development')

# Get a session
with db_manager.get_session() as session:
    # Your database operations here
    users = session.query(User).all()
```

## Troubleshooting

### SQLite Issues
- Ensure you have write permissions in the current directory
- SQLite database file will be created as `llm_hub.db`
- Check for path traversal attempts in custom URLs

### PostgreSQL Issues
- Ensure PostgreSQL is running and accessible
- Check your `DATABASE_URL` format
- Verify database exists: `createdb llm_hub`
- Install PostgreSQL driver: `pip install psycopg2-binary`
- Check hostname validation for remote connections

### General Issues
- Enable SQL debugging: `export SQL_DEBUG=true`
- Check database connection: `python setup_database.py info`
- Use verbose mode for detailed error information: `--verbose`
- Check URL validation errors in logs

## Security Considerations

### SQL Injection Prevention
- All queries use SQLAlchemy ORM or parameterized statements
- No raw SQL with user input is executed
- Database URLs are validated and sanitized

### Access Control
- SQLite databases are created with appropriate file permissions
- PostgreSQL connections use connection pooling with authentication
- Environment variables are validated before use

### Data Protection
- Sensitive information is not logged in error messages
- Database credentials are handled securely
- Connection strings are validated for malicious content

## Dependencies

- `SQLAlchemy>=2.0.0`: ORM framework
- `psycopg2-binary>=2.9.0`: PostgreSQL driver (for production)

SQLite support is built into Python and doesn't require additional drivers. 