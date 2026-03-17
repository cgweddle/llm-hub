# models.py
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON, Table, create_engine, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func
import os
from datetime import datetime
import uuid
import argparse
import sys

Base = declarative_base()

# Association table for many-to-many relationships
agent_tool_association = Table('agent_tool_association', Base.metadata,
    Column('agent_id', Integer, ForeignKey('agents.id')),
    Column('tool_id', Integer, ForeignKey('tools.id'))
)

agent_flow_association = Table('agent_flow_association', Base.metadata,
    Column('agent_id', Integer, ForeignKey('agents.id')),
    Column('flow_id', Integer, ForeignKey('flows.id'))
)

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    agents = relationship("Agent", back_populates="user")
    flows = relationship("Flow", back_populates="user")
    executions = relationship("Execution", back_populates="user")

class Agent(Base):
    __tablename__ = 'agents'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    graph_config = Column(JSON, nullable=False)  # Unified agent workflow graph
    output_schema = Column(JSON)  # JSON schema for structured output validation
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="agents")
    tools = relationship("Tool", secondary=agent_tool_association, back_populates="agents")
    flows = relationship("Flow", secondary=agent_flow_association, back_populates="agents")
    executions = relationship("Execution", back_populates="agent")

class Tool(Base):
    __tablename__ = 'tools'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    tool_type = Column(String(50), nullable=False)  # 'function', 'custom'
    main_function = Column(String(100))  # For function-based tools
    function_code = Column(Text)  # Store actual function code
    helper_functions = Column(JSON)  # Store helper functions as {"name": "code"}
    script_code = Column(Text)  # Store full original Python script text
    input_schema = Column(JSON)  # Input parameters with types and validation
    output_schema = Column(JSON)  # Output structure with types for flow validation
    api_config = Column(JSON)  # For API-based tools
    parameters = Column(JSON)  # Tool parameters schema (legacy, use input_schema)
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    agents = relationship("Agent", secondary=agent_tool_association, back_populates="tools")

class Flow(Base):
    __tablename__ = 'flows'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    graph_config = Column(JSON, nullable=False)  # Store adjacency matrix and node configs
    entry_point = Column(String(100), nullable=False)
    exit_points = Column(JSON)  # List of exit points
    conda_env = Column(String(500), nullable=True)  # Optional conda environment path
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="flows")
    agents = relationship("Agent", secondary=agent_flow_association, back_populates="flows")
    executions = relationship("Execution", back_populates="flow")

class Execution(Base):
    __tablename__ = 'executions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    agent_id = Column(Integer, ForeignKey('agents.id'))
    flow_id = Column(Integer, ForeignKey('flows.id'))
    execution_type = Column(String(20), nullable=False)  # 'agent' or 'flow'
    input_data = Column(JSON)
    output_data = Column(JSON)
    status = Column(String(20), default='running')  # 'running', 'completed', 'failed'
    error_message = Column(Text)
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
    execution_metadata = Column(JSON)  # Additional execution metadata
    
    # Relationships
    user = relationship("User", back_populates="executions")
    agent = relationship("Agent", back_populates="executions")
    flow = relationship("Flow", back_populates="executions")
    messages = relationship("Message", back_populates="execution")

class Message(Base):
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    execution_id = Column(Integer, ForeignKey('executions.id'), nullable=False)
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'tool', 'system'
    content = Column(Text, nullable=False)
    sender = Column(String(100))  # Agent name or tool name
    timestamp = Column(DateTime, default=func.now())
    message_metadata = Column(JSON)  # Additional message metadata
    
    # Relationships
    execution = relationship("Execution", back_populates="messages")

class Prompts(Base):
    __tablename__ = 'prompts'

    id = Column(Integer, primary_key=True)
    prompt_name = Column(Text)
    system_prompt = Column(Text)
    user_prompt = Column(Text)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class DatabaseManager:
    def __init__(self, environment=None):
        """
        Initialize database manager with support for both SQLite and PostgreSQL
        
        Args:
            environment: 'development', 'production', or None (auto-detect)
        """
            # Auto-detect environment if not specified
        if environment is None:
            environment = os.getenv('ENVIRONMENT', 'development')
        
        if environment == 'production':
            # Production defaults to PostgreSQL
            database_url = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/llm_hub')
        else:
            # Development defaults to SQLite
            default_sqlite_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../database/llm_hub.db'))
            os.makedirs(os.path.dirname(default_sqlite_path), exist_ok=True)
            database_url = os.getenv('DATABASE_URL', f'sqlite:///{default_sqlite_path}')
    
        self.database_url = database_url
        self.is_sqlite = self.database_url.startswith('sqlite')
        
        # Configure engine based on database type
        if self.is_sqlite:
            self.engine = create_engine(
                self.database_url, 
                connect_args={
                    "check_same_thread": False,
                    "timeout": 30
                },
                echo=os.getenv('SQL_DEBUG', 'false').lower() == 'true'
            )
        else:
            # PostgreSQL configuration
            self.engine = create_engine(
                self.database_url,
                echo=os.getenv('SQL_DEBUG', 'false').lower() == 'true',
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all tables in the database"""
        try:
            Base.metadata.create_all(bind=self.engine)
            db_type = "SQLite" if self.is_sqlite else "PostgreSQL"
            print(f"Database tables created successfully using {db_type}!")
        except Exception as e:
            print(f"Error creating tables: {e}")
            raise
    
    def get_session(self):
        """Get a database session"""
        return self.SessionLocal()
    
    def close_session(self, session):
        """Close a database session"""
        if session:
            session.close()
    
    def drop_tables(self):
        """Drop all tables in the database (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            db_type = "SQLite" if self.is_sqlite else "PostgreSQL"
            print(f"All tables dropped successfully from {db_type}!")
        except Exception as e:
            print(f"Error dropping tables: {e}")
            raise
    
    def test_connection(self):
        """Test database connection with a simple query"""
        try:
            with self.get_session() as session:
                # Use parameterized query for security
                result = session.execute(text("SELECT 1 as test"))
                result.fetchone()
            return True
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
    
    def get_database_info(self):
        """Get information about the current database configuration"""
        db_type = "SQLite" if self.is_sqlite else "PostgreSQL"
        return {
            "type": db_type,
            "url": self.database_url,
            "is_sqlite": self.is_sqlite
        }


def get_database_manager(environment=None):
    """Get a database manager instance"""
    return DatabaseManager(environment=environment)

def setup_database(environment: str):
    db_manager = get_database_manager(
        environment=environment
    )
    
    print("Creating tables...")
    db_manager.create_tables()
    
    print("Database setup complete!")
    return True


def show_info(environment: str):
    """Show database information"""
    print("=== Database Information ===")
    db_manager = get_database_manager(
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

def drop_database(environment, force):
    """Drop all tables from the database"""
    if not force:
        confirm = input(f"Are you sure you want to drop ALL tables from {environment} environment? This action cannot be undone! (yes/no): ")
        if confirm.lower() != 'yes':
            print("Operation cancelled.")
            return
    
    print(f"Dropping tables from {environment} environment...")
    db_manager = get_database_manager(environment=environment)
    db_manager.drop_tables()
    print("✓ Tables dropped successfully!")

def parse_arguments():
    """Parse command line arguments using argparse"""
    parser = argparse.ArgumentParser(
        description="LLM Hub Database Setup Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python database_setup.py                    # Set up SQLite database (development)
  python database_setup.py development        # Set up SQLite database
  python database_setup.py production         # Set up PostgreSQL database
  python database_setup.py info               # Show database information
  python database_setup.py drop               # Drop all tables (use with caution!)
  python database_setup.py --url sqlite:///custom.db  # Use custom database URL
  python database_setup.py --force            # Skip confirmation prompts
        """
    )
    
    parser.add_argument(
        'action',
        nargs='?',
        default='development',
        choices=['setup', 'info', 'drop'],
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
        if args.action == 'setup':
            setup_database(args.environment)
        elif args.action == 'production':
            setup_database(args.environment)
        elif args.action == 'info':
            show_info(args.environment)
        elif args.action == 'drop':
            drop_database(args.environment, args.force)
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
