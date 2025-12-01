# api.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import sys
import os
import json
import subprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.database.database import (
    get_session, create_agent, get_user_agents, create_user,
    get_available_agents, get_available_tools, get_available_flows,
    get_public_agents, get_public_tools, get_public_flows,
    create_tool, get_user_tools, create_flow, get_user_flows,
    get_tool_by_id, update_tool
)
from src.database.database_setup import DatabaseManager
from src.validate.tool_compatibility import validate_two_tools, validate_tool_compatibility
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime
from passlib.hash import bcrypt

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_manager = DatabaseManager()

# Dependency to get database session
def get_db():
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db_manager.close_session(db)

# Pydantic models for API
class AgentCreate(BaseModel):
    name: str
    description: str
    agent_type: str
    system_prompt: str
    llm_config: dict
    tools_config: dict
    agent_metadata: Optional[dict] = None

class AgentResponse(BaseModel):
    id: int
    name: str
    description: str
    agent_type: str
    created_at: datetime

    class Config:
        from_attributes = True

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ToolCreate(BaseModel):
    name: str
    description: str
    tool_type: str
    function_name: Optional[str] = None
    function_code: Optional[str] = None
    script_code: Optional[str] = None
    input_schema: Optional[dict] = None
    output_schema: Optional[dict] = None
    api_config: Optional[dict] = None
    is_public: bool = False

class ToolResponse(BaseModel):
    id: int
    name: str
    description: str
    tool_type: str
    is_public: bool
    created_at: datetime
    updated_at: datetime
    script_code: Optional[str] = None
    input_schema: Optional[dict] = None
    output_schema: Optional[dict] = None

    class Config:
        from_attributes = True

class ToolUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    script_code: Optional[str] = None
    tool_type: Optional[str] = None
    function_name: Optional[str] = None
    function_code: Optional[str] = None
    input_schema: Optional[dict] = None
    output_schema: Optional[dict] = None
    is_public: Optional[bool] = None

class FlowCreate(BaseModel):
    name: str
    description: str
    graph_config: dict
    entry_point: str
    exit_points: Optional[List[str]] = None
    is_public: bool = False

class FlowResponse(BaseModel):
    id: int
    name: str
    description: str
    is_public: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ValidateTwoToolsRequest(BaseModel):
    tool1_id: int
    tool2_id: int

class ValidateToolChainRequest(BaseModel):
    tool_ids: List[int]

class CondaEnvironment(BaseModel):
    name: str
    path: str

class CondaEnvironmentResponse(BaseModel):
    status: str
    message: str
    environments: List[CondaEnvironment]


@app.post("/agents/", response_model=AgentResponse)
def create_agent_endpoint(agent_data: AgentCreate, user_id: int, db: Session = Depends(get_db)):
    """Create a new agent"""
    try:
        agent = create_agent(
            session=db,
            user_id=user_id,
            name=agent_data.name,
            description=agent_data.description,
            agent_type=agent_data.agent_type,
            system_prompt=agent_data.system_prompt,
            llm_config=agent_data.llm_config,
            tools_config=agent_data.tools_config,
            agent_metadata=agent_data.agent_metadata
        )
        return agent
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")

@app.get("/agents/", response_model=List[AgentResponse])
def get_user_agents_endpoint(user_id: int, db: Session = Depends(get_db)):
    """Get all agents for a user"""
    try:
        return get_user_agents(session=db, user_id=user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user agents: {str(e)}")

@app.post("/agents/{agent_id}/execute")
def execute_agent(agent_id: int, user_id: int, input_data: str, db: Session = Depends(get_db)):
    """Execute an agent"""
    agent_service = AgentService(db)
    # You'll need to get the LLM instance here
    llm = get_llm_instance()  # Implement this based on your LLM setup
    result = agent_service.execute_agent(user_id, agent_id, input_data, llm)
    return {"result": result}

@app.post("/users/", response_model=UserResponse)
def create_user_endpoint(user_data: UserCreate, db: Session = Depends(get_db)):
    """Create a new user"""
    try:
        # Hash the password
        password_hash = bcrypt.hash(user_data.password)
        user = create_user(
            session=db,
            username=user_data.username,
            email=user_data.email,
            password_hash=password_hash
        )
        return user
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")

# Agent endpoints
@app.get("/agents/available/{user_id}", response_model=List[AgentResponse])
def get_available_agents_endpoint(user_id: int, db: Session = Depends(get_db)):
    """Get all agents available to a user (user's agents + public agents)"""
    return get_available_agents(db, user_id)

@app.get("/agents/public", response_model=List[AgentResponse])
def get_public_agents_endpoint(db: Session = Depends(get_db)):
    """Get all public agents"""
    return get_public_agents(db)

# Tool endpoints
@app.post("/tools/", response_model=ToolResponse)
def create_tool_endpoint(tool_data: ToolCreate, user_id: int, db: Session = Depends(get_db)):
    """Create a new tool"""
    tool = create_tool(
        session=db,
        user_id=user_id,
        name=tool_data.name,
        description=tool_data.description,
        tool_type=tool_data.tool_type,
        function_name=tool_data.function_name,
        function_code=tool_data.function_code,
        script_code=tool_data.script_code,
        input_schema=tool_data.input_schema,
        output_schema=tool_data.output_schema,
        is_public=tool_data.is_public
    )
    return tool

@app.get("/tools/available/{user_id}", response_model=List[ToolResponse])
def get_available_tools_endpoint(user_id: int, db: Session = Depends(get_db)):
    """Get all tools available to a user (user's tools + public tools)"""
    return get_available_tools(db, user_id)

@app.get("/tools/public", response_model=List[ToolResponse])
def get_public_tools_endpoint(db: Session = Depends(get_db)):
    """Get all public tools"""
    return get_public_tools(db)

@app.get("/tools/user/{user_id}", response_model=List[ToolResponse])
def get_user_tools_endpoint(user_id: int, db: Session = Depends(get_db)):
    """Get all tools for a specific user"""
    return get_user_tools(db, user_id)

@app.options("/tools/{tool_id}")
async def options_update_tool(tool_id):
    """Handle CORS preflight for PATCH requests on /tools/{tool_id}"""
    # Don't validate tool_id type for OPTIONS - just return empty response
    # CORS middleware will add the necessary headers
    return {}

@app.patch("/tools/{tool_id}", response_model=ToolResponse)
def update_tool_endpoint(tool_id: int, tool_update: ToolUpdate, db: Session = Depends(get_db)):
    """Update a tool's properties (e.g., script_code, description, etc.)"""
    # Get the existing tool
    tool = get_tool_by_id(db, tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")

    # Only update fields that were provided
    update_data = tool_update.model_dump(exclude_unset=True)
    if not update_data:
        return tool

    # Update the tool
    updated_tool = update_tool(db, tool_id, **update_data)
    if not updated_tool:
        raise HTTPException(status_code=500, detail="Failed to update tool")

    return updated_tool

# Tool Validation endpoints
@app.post("/tools/validate-two")
def validate_two_tools_endpoint(request: ValidateTwoToolsRequest):
    """
    Validate if two tools are compatible (tool1 output -> tool2 input)

    Returns:
        - compatible: bool
        - issues: list of compatibility issues
        - compatible_inputs: list of compatible input parameters
        - unsatisfied_required_inputs: list of required inputs that can't be satisfied
        - output_schema: tool1's output schema
        - input_schema: tool2's input schema
    """
    try:
        result = validate_two_tools(request.tool1_id, request.tool2_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/tools/validate-chain")
def validate_tool_chain_endpoint(request: ValidateToolChainRequest):
    """
    Validate a chain of tools in sequence

    Args:
        tool_ids: List of tool IDs in the order they would be chained

    Returns:
        - compatible: bool indicating if all tools are compatible
        - issues: list of compatibility issues found
        - tool_chain: list of tool information with positions
    """
    try:
        result = validate_tool_compatibility(request.tool_ids)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

# Flow endpoints
@app.post("/flows/", response_model=FlowResponse)
def create_flow_endpoint(flow_data: FlowCreate, user_id: int, db: Session = Depends(get_db)):
    """Create a new flow"""
    flow = create_flow(
        session=db,
        user_id=user_id,
        name=flow_data.name,
        description=flow_data.description,
        graph_config=flow_data.graph_config,
        entry_point=flow_data.entry_point,
        exit_points=flow_data.exit_points,
        is_public=flow_data.is_public
    )
    return flow

@app.get("/flows/available/{user_id}", response_model=List[FlowResponse])
def get_available_flows_endpoint(user_id: int, db: Session = Depends(get_db)):
    """Get all flows available to a user (user's flows + public flows)"""
    return get_available_flows(db, user_id)

@app.get("/flows/public", response_model=List[FlowResponse])
def get_public_flows_endpoint(db: Session = Depends(get_db)):
    """Get all public flows"""
    return get_public_flows(db)

@app.get("/flows/user/{user_id}", response_model=List[FlowResponse])
def get_user_flows_endpoint(user_id: int, db: Session = Depends(get_db)):
    """Get all flows for a specific user"""
    return get_user_flows(db, user_id)



# Conda Environment Endpoint
@app.get("/conda/environments", response_model=CondaEnvironmentResponse)
def get_conda_environments():
    """Get available conda environments"""
    try:
        result = subprocess.run(
            ['conda', 'env', 'list', '--json'],
            capture_output=True,
            text=True,
            timeout=10
        )

        conda_info = json.loads(result.stdout)
        envs = conda_info.get('envs', [])

        env_list = []
        for env_path in envs:
            env_name = os.path.basename(env_path)
            env_list.append({"name": env_name, "path": env_path})

        return {
            "status": "success",
            "message": f"Found {len(env_list)} environments",
            "environments": env_list
        }
    except:
        return {"status": "error", "message": "Failed to get conda environments", "environments": []}