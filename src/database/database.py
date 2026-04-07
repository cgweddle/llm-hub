from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker, Session
import os
import logging
from .database_setup import User, Agent, Tool, Flow, Execution, Prompts, Evaluation, EvaluationResult
from dotenv import load_dotenv
load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False  # Prevent propagation to root logger

# Create file handler - ensure logs always go to project root/logs
# Navigate up from scripts/database/ to project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log_file_path = os.path.join(project_root, 'logs', 'database.log')

# Ensure logs directory exists
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(file_handler)

# Test the logger
logger.debug("DEBUG: Database module loaded and logger initialized")
logger.info("INFO: Database module loaded and logger initialized")



# Setup database session
def get_session():
    # Debug: Check if logger is working
    print(f"Logger name: {logger.name}")
    print(f"Logger level: {logger.level}")
    print(f"Logger handlers: {len(logger.handlers)}")
    print(f"Logger propagate: {logger.propagate}")
    
    logger.debug("getting environment variable")
    database_url = os.getenv('DATABASE_URL')
    logger.info(f"Attempting to connect to database with URL: {database_url}")
    
    if not database_url:
        logger.error("DATABASE_URL environment variable is not set")
        raise ValueError("DATABASE_URL environment variable is not set")
    
    # For SQLite, we need check_same_thread=False
    if database_url.startswith('sqlite'):
        logger.info("Creating SQLite engine with check_same_thread=False")
        # Parse URL so we can grab the DB path
        url = make_url(database_url)

        # url.database is the path part, e.g. "llm_hub/database/llm_hub"
        db_path = url.database

        # If it's not absolute, make it relative to project_root
        if not os.path.isabs(db_path):
            db_path = os.path.join(project_root, db_path)

        fixed_url = f"sqlite:///{db_path}"
        logger.info(f"Using SQLite DB path: {db_path}")
        logger.info(f"Final SQLite URL: {fixed_url}")

        engine = create_engine(
            fixed_url,
            connect_args={"check_same_thread": False}
        )
    else:
        logger.info("Creating database engine")
        engine = create_engine(database_url)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("Database session created successfully")
    return SessionLocal()


## Association table sync helpers

def _sync_agent_tools(session: Any, agent: Agent, graph_config: Dict):
    """Sync agent_tool_association from tool_ids in graph_config nodes."""
    tool_ids = set()
    for node in graph_config.get("nodes", {}).values():
        tool_ids.update(node.get("tool_ids", []))
    agent.tools = session.query(Tool).filter(Tool.id.in_(tool_ids)).all() if tool_ids else []

def _sync_flow_associations(session: Any, flow: Flow, graph_config: Dict):
    """Sync agent_flow_association and flow_tool_association from graph_config nodes."""
    agent_ids = set()
    tool_ids = set()
    for node in graph_config.get("nodes", {}).values():
        if node.get("node_type") == "agent":
            if agent_id := node.get("agent_id"):
                agent_ids.add(agent_id)
        elif node.get("node_type") != "trigger":
            tool_id = node.get("tool_id") or node.get("id")
            if tool_id:
                tool_ids.add(tool_id)
    flow.agents = session.query(Agent).filter(Agent.id.in_(agent_ids)).all() if agent_ids else []
    flow.tools = session.query(Tool).filter(Tool.id.in_(tool_ids)).all() if tool_ids else []


## Agent database

def create_agent(session: Any, user_id: int, name: str, description: str,
                    graph_config: Dict, output_schema: Dict = None) -> Agent:
    """Create a new agent with a unified graph_config.

    graph_config contains nodes, edges, entry_point, exit_points, and
    optionally max_loop_iterations. Every agent — simple or complex —
    is represented as a graph.
    """
    agent = Agent(
        user_id=user_id,
        name=name,
        description=description,
        graph_config=graph_config,
        output_schema=output_schema
    )
    session.add(agent)
    _sync_agent_tools(session, agent, graph_config)
    session.commit()
    return agent

def get_user_agents(session: Any, user_id: int) -> List[Agent]:
    """Get all agents for a user"""
    return session.query(Agent).filter(Agent.user_id == user_id).all()

def get_agent_by_id(session: Any, agent_id: int) -> Optional[Agent]:
    """Get agent by ID"""
    return session.query(Agent).filter(Agent.id == agent_id).first()

def update_agent(session: Any, agent_id: int, **kwargs) -> Optional[Agent]:
    """Update an agent"""
    agent = get_agent_by_id(session, agent_id)
    if agent:
        for key, value in kwargs.items():
            setattr(agent, key, value)
        if "graph_config" in kwargs:
            _sync_agent_tools(session, agent, kwargs["graph_config"])
        session.commit()
        session.refresh(agent)
    return agent

def delete_agent(session: Any, agent_id: int) -> bool:
    """Delete an agent"""
    agent = get_agent_by_id(session, agent_id)
    if agent:
        session.delete(agent)
        session.commit()
        return True
    return False


## Flow database
def create_flow(session: Any, user_id: int, name: str, description: str,
                graph_config: Dict, is_public: bool = False, conda_env: str = None) -> Flow:
    """Create a new flow"""
    flow = Flow(
        user_id=user_id,
        name=name,
        description=description,
        graph_config=graph_config,
        entry_point=graph_config.get("entry_point", "START"),
        exit_points=graph_config.get("exit_points", []),
        conda_env=conda_env,
        is_public=is_public
    )
    session.add(flow)
    _sync_flow_associations(session, flow, graph_config)
    session.commit()
    return flow
    
def get_user_flows(session: Any, user_id: int) -> List[Flow]:
    """Get all flows for a user"""
    return session.query(Flow).filter(Flow.user_id == user_id).all()

def get_flow_by_id(session: Any, flow_id: int) -> Optional[Flow]:
    """Get flow by ID"""
    return session.query(Flow).filter(Flow.id == flow_id).first()

def update_flow(session: Any, flow_id: int, **kwargs) -> Optional[Flow]:
    """Update a flow"""
    flow = get_flow_by_id(session, flow_id)
    if flow:
        for key, value in kwargs.items():
            setattr(flow, key, value)
        if "graph_config" in kwargs:
            _sync_flow_associations(session, flow, kwargs["graph_config"])
        session.commit()
    return flow

def delete_flow(session: Any, flow_id: int) -> bool:
    """Delete a flow"""
    flow = get_flow_by_id(session, flow_id)
    if flow:
        session.delete(flow)
        session.commit()
        return True
    return False

def create_user(session: Any, username: str, email: str, password_hash: str, is_active: bool = True) -> User:
    """Create a new user"""
    logger.info(f"Creating new user: {username} ({email})")
    
    user = User(
        username=username,
        email=email,
        password_hash=password_hash,
        is_active=is_active
    )
    session.add(user)
    session.commit()
    logger.info(f"User created successfully with ID: {user.id}")
    return user

## Tool database functions
def create_tool(session: Any, user_id: int, name: str, description: str,
                tool_type: str, main_function: str = None, function_code: str = None,
                helper_functions: Dict = None, script_code: str = None, input_schema: Dict = None,
                output_schema: Dict = None, api_config: Dict = None, is_public: bool = False) -> Tool:
    """Create a new tool with helper functions and type schemas"""
    tool = Tool(
        user_id=user_id,
        name=name,
        description=description,
        tool_type=tool_type,
        main_function=main_function,
        function_code=function_code,
        script_code=script_code,
        input_schema=input_schema or {},
        output_schema=output_schema or {},
        is_public=is_public
    )
    session.add(tool)
    session.commit()
    return tool

def get_user_tools(session: Any, user_id: int) -> List[Tool]:
    """Get all tools for a user"""
    return session.query(Tool).filter(Tool.user_id == user_id).all()

def get_tool_by_id(session: Any, tool_id: int) -> Optional[Tool]:
    """Get tool by ID"""
    return session.query(Tool).filter(Tool.id == tool_id).first()

def update_tool(session: Any, tool_id: int, **kwargs) -> Optional[Tool]:
    """Update a tool"""
    tool = get_tool_by_id(session, tool_id)
    if tool:
        for key, value in kwargs.items():
            setattr(tool, key, value)
        session.commit()
    return tool

def delete_tool(session: Any, tool_id: int) -> bool:
    """Delete a tool"""
    tool = get_tool_by_id(session, tool_id)
    if tool:
        session.delete(tool)
        session.commit()
        return True
    return False


## Public items functions
def get_public_agents(session: Any) -> List[Agent]:
    """Get all public agents"""
    return session.query(Agent).filter(Agent.is_public == True).all()

def get_public_tools(session: Any) -> List[Tool]:
    """Get all public tools"""
    return session.query(Tool).filter(Tool.is_public == True).all()

def get_public_flows(session: Any) -> List[Flow]:
    """Get all public flows"""
    return session.query(Flow).filter(Flow.is_public == True).all()

def get_available_agents(session: Any, user_id: int) -> List[Agent]:
    """Get agents available to a user (user's agents + public agents)"""
    user_agents = get_user_agents(session, user_id)
    public_agents = get_public_agents(session)
    
    # Combine and deduplicate
    all_agents = user_agents + public_agents
    seen_ids = set()
    unique_agents = []
    for agent in all_agents:
        if agent.id not in seen_ids:
            seen_ids.add(agent.id)
            unique_agents.append(agent)
    
    return unique_agents

def get_available_tools(session: Any, user_id: int) -> List[Tool]:
    """Get tools available to a user (user's tools + public tools)"""
    user_tools = get_user_tools(session, user_id)
    public_tools = get_public_tools(session)
    
    # Combine and deduplicate
    all_tools = user_tools + public_tools
    seen_ids = set()
    unique_tools = []
    for tool in all_tools:
        if tool.id not in seen_ids:
            seen_ids.add(tool.id)
            unique_tools.append(tool)
    
    return unique_tools

def get_available_flows(session: Any, user_id: int) -> List[Flow]:
    """Get flows available to a user (user's flows + public flows)"""
    user_flows = get_user_flows(session, user_id)
    public_flows = get_public_flows(session)
    
    # Combine and deduplicate
    all_flows = user_flows + public_flows
    seen_ids = set()
    unique_flows = []
    for flow in all_flows:
        if flow.id not in seen_ids:
            seen_ids.add(flow.id)
            unique_flows.append(flow)

    return unique_flows

def get_prompt_by_name(session: Any, prompt_name: str) -> Optional[Prompts]:
    """Get prompt by name from Prompts table"""
    return session.query(Prompts).filter(Prompts.prompt_name == prompt_name).first()


## Execution database functions

def create_execution(session: Any, **kwargs) -> Execution:
    """Create an execution record. Accepts any Execution column as a kwarg."""
    execution = Execution(**kwargs)
    session.add(execution)
    session.commit()
    session.refresh(execution)
    return execution

def get_execution_by_id(session: Any, execution_id: int) -> Optional[Execution]:
    """Get a single execution by ID (children are lazy-loaded via relationship)."""
    return session.query(Execution).filter(Execution.id == execution_id).first()

def get_user_executions(session: Any, user_id: int, limit: int = 50, offset: int = 0) -> List[Execution]:
    """Get top-level executions for a user (parent_id is NULL), newest first."""
    return (
        session.query(Execution)
        .filter(Execution.user_id == user_id, Execution.parent_id.is_(None))
        .order_by(Execution.started_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

def update_execution(session: Any, execution_id: int, **kwargs) -> Optional[Execution]:
    """Update an execution record."""
    execution = get_execution_by_id(session, execution_id)
    if execution:
        for key, value in kwargs.items():
            setattr(execution, key, value)
        session.commit()
        session.refresh(execution)
    return execution


## Evaluation database functions

def create_evaluation(session: Any, **kwargs) -> Evaluation:
    evaluation = Evaluation(**kwargs)
    session.add(evaluation)
    session.commit()
    session.refresh(evaluation)
    return evaluation

def get_evaluations_by_user(session: Any, user_id: int) -> List[Evaluation]:
    user_evals = session.query(Evaluation).filter(Evaluation.user_id == user_id).all()
    public_evals = session.query(Evaluation).filter(Evaluation.is_public == True, Evaluation.user_id != user_id).all()
    return user_evals + public_evals

def get_evaluation_by_id(session: Any, evaluation_id: int) -> Optional[Evaluation]:
    return session.query(Evaluation).filter(Evaluation.id == evaluation_id).first()

def update_evaluation(session: Any, evaluation_id: int, **kwargs) -> Optional[Evaluation]:
    evaluation = get_evaluation_by_id(session, evaluation_id)
    if evaluation:
        for key, value in kwargs.items():
            setattr(evaluation, key, value)
        session.commit()
        session.refresh(evaluation)
    return evaluation

def delete_evaluation(session: Any, evaluation_id: int) -> bool:
    evaluation = get_evaluation_by_id(session, evaluation_id)
    if evaluation:
        session.delete(evaluation)
        session.commit()
        return True
    return False


## Evaluation Result database functions

def create_evaluation_result(session: Any, **kwargs) -> EvaluationResult:
    result = EvaluationResult(**kwargs)
    session.add(result)
    session.commit()
    session.refresh(result)
    return result

def get_evaluation_results_by_execution(session: Any, execution_id: int) -> List[EvaluationResult]:
    return session.query(EvaluationResult).filter(EvaluationResult.execution_id == execution_id).all()

def update_evaluation_result(session: Any, result_id: int, **kwargs) -> Optional[EvaluationResult]:
    result = session.query(EvaluationResult).filter(EvaluationResult.id == result_id).first()
    if result:
        for key, value in kwargs.items():
            setattr(result, key, value)
        session.commit()
        session.refresh(result)
    return result