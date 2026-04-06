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
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

# Association table for many-to-many relationships
agent_tool_association = Table('agent_tool_association', Base.metadata,
    Column('agent_id', Integer, ForeignKey('agents.id', ondelete="CASCADE")),
    Column('tool_id', Integer, ForeignKey('tools.id', ondelete="CASCADE"))
)

agent_flow_association = Table('agent_flow_association', Base.metadata,
    Column('agent_id', Integer, ForeignKey('agents.id', ondelete="CASCADE")),
    Column('flow_id', Integer, ForeignKey('flows.id', ondelete="CASCADE"))
)

flow_tool_association = Table('flow_tool_association', Base.metadata,
    Column('flow_id', Integer, ForeignKey('flows.id', ondelete="CASCADE")),
    Column('tool_id', Integer, ForeignKey('tools.id', ondelete="CASCADE"))
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
    flows = relationship("Flow", secondary=flow_tool_association, back_populates="tools")
    executions = relationship("Execution", back_populates="tool")

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
    tools = relationship("Tool", secondary=flow_tool_association, back_populates="flows")
    executions = relationship("Execution", back_populates="flow")

class Execution(Base):
    """
    Self-referencing execution tree. Every execution — flow, agent, tool,
    tool_call, tool_result, trigger — is a row in this table.
    Top-level executions have parent_id=NULL. Children reference their parent.
    """
    __tablename__ = 'executions'

    id = Column(Integer, primary_key=True)
    parent_id = Column(Integer, ForeignKey('executions.id'), nullable=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    agent_id = Column(Integer, ForeignKey('agents.id'), nullable=True)
    flow_id = Column(Integer, ForeignKey('flows.id'), nullable=True)
    tool_id = Column(Integer, ForeignKey('tools.id'), nullable=True)
    execution_type = Column(String(50), nullable=False)  # 'flow', 'agent', 'tool', 'tool_call', 'tool_result', 'trigger'
    node_id = Column(String(100), nullable=True)  # Node identifier from graph_config
    name = Column(String(200), nullable=True)  # Human-readable name
    sequence = Column(Integer, nullable=True)  # Execution order within parent
    input_data = Column(JSON)
    output_data = Column(JSON)
    status = Column(String(20), default='running')  # 'running', 'completed', 'failed'
    error_message = Column(Text)
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
    execution_metadata = Column(JSON)  # Cost, model name, token counts, etc.
    langfuse_trace_id = Column(String(200), nullable=True)  # LangFuse trace ID for cross-referencing

    # Self-referential relationships
    parent = relationship("Execution", remote_side=[id], back_populates="children")
    children = relationship("Execution", back_populates="parent", order_by="Execution.sequence")

    # Foreign key relationships
    user = relationship("User", back_populates="executions")
    agent = relationship("Agent", back_populates="executions")
    flow = relationship("Flow", back_populates="executions")
    tool = relationship("Tool", back_populates="executions")

class Prompts(Base):
    __tablename__ = 'prompts'

    id = Column(Integer, primary_key=True)
    prompt_name = Column(Text)
    system_prompt = Column(Text)
    user_prompt = Column(Text)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class DatabaseManager:
    def __init__(self):
        """
        Initialize database manager with support for both SQLite and PostgreSQL.
        Set DATABASE_URL to a PostgreSQL connection string for production,
        or leave it unset to default to local SQLite.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_sqlite_path = os.path.join(project_root, 'database', 'llm_hub.db')
        database_url = os.getenv('DATABASE_URL', f'sqlite:///{default_sqlite_path}')
        if database_url.startswith('sqlite'):
            os.makedirs(os.path.dirname(default_sqlite_path), exist_ok=True)
    
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


def get_database_manager():
    """Get a database manager instance"""
    return DatabaseManager()

def setup_database():
    db_manager = get_database_manager()

    print("Creating tables...")
    db_manager.create_tables()

    print("Database setup complete!")
    return True


def show_info():
    """Show database information"""
    print("=== Database Information ===")
    db_manager = get_database_manager()
    info = db_manager.get_database_info()

    print(f"Database Type: {info['type']}")
    print(f"Database URL: {info['url']}")

    if db_manager.is_sqlite:
        db_file = db_manager.database_url.replace('sqlite:///', '')
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

def drop_database(force):
    """Drop all tables from the database"""
    db_manager = get_database_manager()
    db_type = "SQLite" if db_manager.is_sqlite else "PostgreSQL"
    if not force:
        confirm = input(f"Are you sure you want to drop ALL tables from {db_type} database? This action cannot be undone! (yes/no): ")
        if confirm.lower() != 'yes':
            print("Operation cancelled.")
            return

    print(f"Dropping tables from {db_type} database...")
    db_manager.drop_tables()
    print("✓ Tables dropped successfully!")

def parse_arguments():
    """Parse command line arguments using argparse"""
    parser = argparse.ArgumentParser(
        description="LLM Hub Database Setup Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python database_setup.py setup              # Create tables (uses DATABASE_URL)
  python database_setup.py info               # Show database information
  python database_setup.py drop               # Drop all tables (use with caution!)
  python database_setup.py drop --force       # Skip confirmation prompts
        """
    )

    parser.add_argument(
        'action',
        nargs='?',
        default='setup',
        choices=['setup', 'info', 'drop'],
        help='Action to perform (default: setup)'
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
            print(f"DATABASE_URL: {os.getenv('DATABASE_URL', '(not set, using SQLite default)')}")

        # Handle different actions
        if args.action == 'setup':
            setup_database()
        elif args.action == 'info':
            show_info()
        elif args.action == 'drop':
            drop_database(args.force)
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
