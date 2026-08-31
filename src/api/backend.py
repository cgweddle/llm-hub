# api.py
import sys
import io
import os
import json
import subprocess
import time
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from passlib.hash import bcrypt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.database.database import (
    get_session, create_agent, get_user_agents, create_user,
    get_available_agents, get_available_tools, get_available_flows,
    get_public_agents, get_public_tools, get_public_flows,
    create_tool, get_user_tools, create_flow, get_user_flows,
    get_tool_by_id, update_tool, delete_tool, get_flow_by_id, update_flow, delete_flow,
    get_agent_by_id, update_agent, delete_agent,
    get_execution_by_id, get_user_executions, create_execution, update_execution,
    create_evaluation, get_evaluations_by_user, get_evaluation_by_id,
    update_evaluation, delete_evaluation,
    create_evaluation_result, get_evaluation_results_by_execution,
)
from src.utils import load_llm_provider_config, save_llm_provider_config
from src.utils.environment import is_hosted
from src.database.database import get_user_llm_provider_configs, sync_user_llm_configs
from src.database.database_setup import DatabaseManager
from src.validate.tool_compatibility import validate_two_tools, validate_tool_compatibility, validate_connection
from src.runners.local_flow_child import spawn_local_flow_child
from src.runners.live_run_store import live_run_store
from src.factories.python_script_tool_factory import PythonScriptToolFactory
from src.factories.pigar_import_detector import detect_required_packages
from src.exporters.flow_exporter import FlowExportError, export_flow_zip


def load_request_llm_config(user_id: Optional[int], session=None) -> Dict[str, Any]:
    """Load the `{"models": [...]}` config for a request.

    HOSTED: reads decrypted per-user rows from llm_provider_configs.
    LOCAL:  reads ~/.llm_hub/config.yaml.
    """
    if is_hosted() and user_id is not None:
        owns_session = session is None
        if owns_session:
            session = db_manager.get_session()
        try:
            return {"models": get_user_llm_provider_configs(session, user_id)}
        finally:
            if owns_session:
                session.close()
    return load_llm_provider_config()


def _resolve_llm_config(model_name: str, user_id: Optional[int] = None) -> Dict[str, Any]:
    """Resolve a single LLM config dict by name for the given user."""
    config = load_request_llm_config(user_id)
    for m in config.get("models", []):
        if m.get("name") == model_name:
            return m
    raise ValueError(f"LLM config '{model_name}' not found")

app = FastAPI()

# Add CORS middleware
cors_origins_raw = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:5174,"
    "http://127.0.0.1:3000,http://127.0.0.1:5173,http://127.0.0.1:5174"
)
cors_origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_manager = DatabaseManager()

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

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
    graph_config: dict              # Required — unified agent graph
    output_schema: Optional[dict] = None
    is_public: bool = False

class AgentResponse(BaseModel):
    id: int
    name: str
    description: str
    graph_config: dict
    output_schema: Optional[dict] = None
    is_public: bool = False
    created_at: datetime

    class Config:
        from_attributes = True

class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    graph_config: Optional[dict] = None
    output_schema: Optional[dict] = None
    is_public: Optional[bool] = None

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
    main_function: Optional[str] = None
    script_code: Optional[str] = None
    input_schema: Optional[dict] = None
    output_schema: Optional[dict] = None
    required_packages: Optional[List[str]] = None

    class Config:
        from_attributes = True

class ToolUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    script_code: Optional[str] = None
    tool_type: Optional[str] = None
    main_function: Optional[str] = None
    function_code: Optional[str] = None
    input_schema: Optional[dict] = None
    output_schema: Optional[dict] = None
    is_public: Optional[bool] = None

class FlowCreate(BaseModel):
    name: str
    description: str
    graph_config: dict
    is_public: bool = False
    conda_env: Optional[str] = None

class FlowResponse(BaseModel):
    id: int
    name: str
    description: str
    is_public: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class FlowUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    graph_config: Optional[dict] = None
    is_public: Optional[bool] = None
    conda_env: Optional[str] = None

class ValidateTwoToolsRequest(BaseModel):
    tool1_id: int
    tool2_id: int

class ValidateSpecificConnectionRequest(BaseModel):
    tool1_id: int
    tool2_id: int
    source_field: str = ""
    target_field: str = ""

class ValidateToolChainRequest(BaseModel):
    tool_ids: List[int]

class CondaEnvironment(BaseModel):
    name: str
    path: str

class CondaEnvironmentResponse(BaseModel):
    status: str
    message: str
    environments: List[CondaEnvironment]

class FlowExecuteRequest(BaseModel):
    user_id: int
    initial_input: dict
    conda_env: Optional[str] = None
    agent_llms: Dict[str, str] = {}

class FlowExportRequest(BaseModel):
    user_id: int = 1
    agent_llms: Dict[str, str] = {}

class ExecutionResponse(BaseModel):
    """Recursive execution tree node."""
    id: int
    parent_id: Optional[int] = None
    execution_type: str
    node_id: Optional[str] = None
    name: Optional[str] = None
    sequence: Optional[int] = None
    input_data: Optional[Any] = None
    output_data: Optional[Any] = None
    status: str
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_metadata: Optional[Dict[str, Any]] = None
    children: List["ExecutionResponse"] = []

    class Config:
        from_attributes = True

# Needed for the self-referencing model
ExecutionResponse.model_rebuild()

class ExecutionListItem(BaseModel):
    """Lightweight execution summary for list endpoints."""
    id: int
    execution_type: str
    name: Optional[str] = None
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    flow_id: Optional[int] = None
    agent_id: Optional[int] = None

    class Config:
        from_attributes = True

class PythonScriptToolCreate(BaseModel):
    name: str
    description: str
    script_code: str
    main_function: str
    is_public: bool = False

class SystemPromptGenerateRequest(BaseModel):
    agent_name: str
    agent_description: str
    tool_names: List[str] = []
    model: str
    additional_instructions: Optional[str] = None
    user_id: Optional[int] = None

class UserPromptGenerateRequest(BaseModel):
    agent_name: str
    agent_description: str
    tool_names: List[str] = []
    model: str
    generated_system_prompt: str
    additional_instructions: Optional[str] = None
    user_id: Optional[int] = None


class CodeGenerateRequest(BaseModel):
    tool_name: str
    tool_description: str
    model: str
    additional_instructions: Optional[str] = None
    user_id: Optional[int] = None

class CodeGenerateResponse(BaseModel):
    script_code: str
    main_function: str

class CodeEditRequest(BaseModel):
    existing_code: str
    editing_instructions: str
    tool_name: str
    tool_description: str
    model: str
    user_id: Optional[int] = None

class LLMProviderConfig(BaseModel):
    name: str
    provider: str  # 'anthropic' | 'openai' | 'gemini' | 'lmstudio'
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str

class LLMProvidersConfigRequest(BaseModel):
    models: List[LLMProviderConfig]

class LLMProvidersConfigResponse(BaseModel):
    models: List[Dict[str, Any]]


# --- Evaluation models ---

class EvaluationCreate(BaseModel):
    name: str
    description: Optional[str] = None
    judge_system_prompt: str
    judge_user_prompt: Optional[str] = None
    scoring_rubric: Optional[str] = None
    score_type: str  # 'NUMERIC', 'CATEGORICAL', 'BOOLEAN'
    score_categories: Optional[List[Dict[str, Any]]] = None  # [{"name": "good", "description": "..."}]
    llm_provider: str
    input_variables: Optional[List[str]] = None  # ["output", "input", "tool_output"]
    return_fields: Optional[List[str]] = None  # ["reasoning", "confidence"]
    is_public: bool = False

class EvaluationUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    judge_system_prompt: Optional[str] = None
    judge_user_prompt: Optional[str] = None
    scoring_rubric: Optional[str] = None
    score_type: Optional[str] = None
    score_categories: Optional[List[Dict[str, Any]]] = None
    llm_provider: Optional[str] = None
    input_variables: Optional[List[str]] = None
    return_fields: Optional[List[str]] = None
    is_public: Optional[bool] = None

class EvaluationResponse(BaseModel):
    id: int
    user_id: int
    name: str
    description: Optional[str]
    judge_system_prompt: str
    judge_user_prompt: Optional[str]
    scoring_rubric: Optional[str]
    score_type: str
    score_categories: Optional[List[Dict[str, Any]]]
    llm_provider: str
    input_variables: Optional[List[str]]
    return_fields: Optional[List[str]]
    is_public: bool
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True

class EvaluateRequest(BaseModel):
    user_id: int
    evaluation_ids: List[int]
    llm_provider: Optional[str] = None  # Optional override; defaults to evaluation's llm_provider

class EvaluationResultResponse(BaseModel):
    id: int
    evaluation_id: int
    evaluation_name: Optional[str] = None
    execution_id: int
    langfuse_trace_id: Optional[str]
    langfuse_score_id: Optional[str]
    status: str
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]
    class Config:
        from_attributes = True


@app.post("/agents/", response_model=AgentResponse)
def create_agent_endpoint(agent_data: AgentCreate, user_id: int, db: Session = Depends(get_db)):
    """Create a new agent"""
    try:
        agent = create_agent(
            session=db,
            user_id=user_id,
            name=agent_data.name,
            description=agent_data.description,
            graph_config=agent_data.graph_config,
            output_schema=agent_data.output_schema
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

@app.patch("/agents/{agent_id}", response_model=AgentResponse)
def update_agent_endpoint(agent_id: int, agent_update: AgentUpdate, db: Session = Depends(get_db)):
    """Update an agent's properties"""
    agent = get_agent_by_id(db, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    update_data = agent_update.model_dump(exclude_unset=True)
    if not update_data:
        return agent

    try:
        updated_agent = update_agent(db, agent_id, **update_data)
        return updated_agent
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update agent: {str(e)}")

@app.delete("/agents/{agent_id}")
def delete_agent_endpoint(agent_id: int, db: Session = Depends(get_db)):
    """Delete an agent"""
    try:
        success = delete_agent(db, agent_id)
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        return {"status": "success", "message": f"Agent {agent_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent deletion failed: {str(e)}")

@app.post("/agents/generate-system-prompt")
async def generate_system_prompt_endpoint(request: SystemPromptGenerateRequest, db: Session = Depends(get_db)):
    """Generate a system prompt for an agent using AI with streaming"""
    try:
        from src.ai_integrations.generate_agent_system_prompt import generate_system_prompt_stream

        llm_config = _resolve_llm_config(request.model, request.user_id)

        async def stream_generator():
            try:
                async for chunk in generate_system_prompt_stream(
                    session=db,
                    agent_name=request.agent_name,
                    agent_description=request.agent_description,
                    tool_names=request.tool_names,
                    llm_config=llm_config,
                    additional_instructions=request.additional_instructions,
                ):
                    yield chunk
            except Exception as e:
                yield json.dumps({"error": f"Streaming error: {str(e)}"}) + "\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System prompt generation failed: {str(e)}")

@app.post("/agents/generate-user-prompt")
async def generate_user_prompt_endpoint(request: UserPromptGenerateRequest, db: Session = Depends(get_db)):
    """Generate a task-specific user prompt for an agent using AI with streaming"""
    try:
        from src.ai_integrations.generate_agent_system_prompt import generate_user_prompt_stream
        llm_config = _resolve_llm_config(request.model, request.user_id)

        async def stream_generator():
            try:
                async for chunk in generate_user_prompt_stream(
                    session=db,
                    agent_name=request.agent_name,
                    agent_description=request.agent_description,
                    tool_names=request.tool_names,
                    llm_config=llm_config,
                    generated_system_prompt=request.generated_system_prompt,
                    additional_instructions=request.additional_instructions,
                ):
                    yield chunk
            except Exception as e:
                yield json.dumps({"error": f"Streaming error: {str(e)}"}) + "\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User prompt generation failed: {str(e)}")

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

@app.post("/tools/python-script", response_model=ToolResponse)
def create_python_script_tool_endpoint(tool_data: PythonScriptToolCreate, user_id: int, db: Session = Depends(get_db)):
    """Create a new tool from a Python script using PythonScriptToolFactory"""
    try:
        # Create factory with the specified main function
        factory = PythonScriptToolFactory(main_function=tool_data.main_function)
        
        # Create tool using the factory
        tool_id = factory.create_tool_from_script(
            script_code=tool_data.script_code,
            tool_name=tool_data.name,
            tool_description=tool_data.description,
            user_id=user_id,
            is_public=tool_data.is_public
        )

        # Fetch the created tool
        created_tool = get_tool_by_id(db, tool_id)
        if not created_tool:
            raise HTTPException(status_code=500, detail="Tool was created but could not be retrieved")

        return created_tool
        
    except ValueError as e:
        # Factory validation errors (e.g., invalid syntax, missing function)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Other errors
        raise HTTPException(status_code=500, detail=f"Failed to create python script tool: {str(e)}")

@app.post("/tools/generate-code")
async def generate_tool_code_endpoint(request: CodeGenerateRequest, db: Session = Depends(get_db)):
    """Generate Python tool code using AI with user-selected LLM (streaming)"""
    try:
        from src.ai_integrations.generate_python_tools import generate_tool_code_stream
        llm_config = _resolve_llm_config(request.model, request.user_id)

        async def stream_generator():
            try:
                async for chunk in generate_tool_code_stream(
                    session=db,
                    tool_name=request.tool_name,
                    tool_description=request.tool_description,
                    llm_config=llm_config,
                    additional_instructions=request.additional_instructions,
                ):
                    yield chunk
            except Exception as e:
                yield json.dumps({"error": f"Streaming error: {str(e)}"}) + "\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    except ValueError as e:
        # Missing prompts or validation errors
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # LLM invocation or other errors
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")

@app.post("/tools/edit-code")
async def edit_tool_code_endpoint(request: CodeEditRequest, db: Session = Depends(get_db)):
    """Edit existing Python tool code using AI with user-selected LLM (streaming)"""
    try:
        from src.ai_integrations.generate_python_tools import edit_tool_code_stream
        llm_config = _resolve_llm_config(request.model, request.user_id)

        async def stream_generator():
            try:
                async for chunk in edit_tool_code_stream(
                    session=db,
                    existing_code=request.existing_code,
                    editing_instructions=request.editing_instructions,
                    tool_name=request.tool_name,
                    tool_description=request.tool_description,
                    llm_config=llm_config,
                ):
                    yield chunk
            except Exception as e:
                yield json.dumps({"error": f"Streaming error: {str(e)}"}) + "\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    except ValueError as e:
        # Missing prompts or validation errors
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # LLM invocation or other errors
        raise HTTPException(status_code=500, detail=f"Code editing failed: {str(e)}")

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

@app.get("/tools/{tool_id}", response_model=ToolResponse)
def get_tool_endpoint(tool_id: int, db: Session = Depends(get_db)):
    """Get a single tool by ID"""
    tool = get_tool_by_id(db, tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    return tool

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

    # If script_code or main_function changed, regenerate schemas
    if 'script_code' in update_data or 'main_function' in update_data:
        script_code = update_data.get('script_code', tool.script_code)
        main_function = update_data.get('main_function', tool.main_function)

        if script_code and main_function:
            try:
                # Regenerate schemas using the factory
                factory = PythonScriptToolFactory()
                analyzer = factory.analyzer
                functions = analyzer.parse_script(script_code)

                if main_function in functions:
                    schema_gen = factory.schema_generator
                    update_data['input_schema'] = schema_gen.generate_input_schema(functions[main_function])
                    update_data['output_schema'] = schema_gen.generate_output_schema(functions[main_function])
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to parse script: {str(e)}")

        if 'script_code' in update_data and script_code:
            update_data['required_packages'] = detect_required_packages(script_code)

    # Update the tool
    updated_tool = update_tool(db, tool_id, **update_data)
    if not updated_tool:
        raise HTTPException(status_code=500, detail="Failed to update tool")

    return updated_tool

@app.delete("/tools/{tool_id}")
def delete_tool_endpoint(tool_id: int, db: Session = Depends(get_db)):
    """Delete a tool"""
    try:
        success = delete_tool(db, tool_id)
        if not success:
            raise HTTPException(status_code=404, detail="Tool not found")
        return {"status": "success", "message": f"Tool {tool_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tool deletion failed: {str(e)}")

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

@app.post("/tools/validate-connection")
def validate_connection_endpoint(request: ValidateSpecificConnectionRequest):
    """
    Validate a specific field-to-field connection between two tools

    Returns:
        - compatible: bool
        - source_field: name of source field
        - target_field: name of target field
        - source_type: type of the source field
        - target_type: type of the target field
    """
    try:
        result = validate_connection(
            request.tool1_id,
            request.tool2_id,
            request.source_field,
            request.target_field
        )
        return {
            "compatible": result["compatible"],
            "source_field": result["source_field"],
            "target_field": result["target_field"],
            "source_type": result["source_type"],
            "target_type": result["target_type"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Connection validation failed: {str(e)}")

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
        is_public=flow_data.is_public,
        conda_env=flow_data.conda_env
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

@app.get("/flows/{flow_id}")
def get_flow_endpoint(flow_id: int, db: Session = Depends(get_db)):
    """Get a single flow with full details including graph_config"""
    flow = get_flow_by_id(db, flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")

    return {
        "id": flow.id,
        "name": flow.name,
        "description": flow.description,
        "graph_config": flow.graph_config,
        "conda_env": flow.conda_env,
        "is_public": flow.is_public,
        "created_at": flow.created_at,
        "updated_at": flow.updated_at
    }

@app.post("/flows/{flow_id}/export")
def export_flow_endpoint(flow_id: int, request: FlowExportRequest, db: Session = Depends(get_db)):
    """Export a flow as a standalone Python module (zip download).

    agent_llms maps agent node ids to LLM config names — the same selection
    the execute endpoint takes; providers/models are baked into the export
    but credentials are read from env vars at run time, never embedded.
    """
    flow = get_flow_by_id(db, flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    try:
        llm_config = load_request_llm_config(request.user_id, session=db)
        zip_bytes, filename = export_flow_zip(flow, db, llm_config, request.agent_llms)
    except FlowExportError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flow export failed: {str(e)}")
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

def _watch_local_flow_child(child, execution_id: int, flow_id: int,
                            prev_completed_at=None) -> dict:
    """Sync facade over the per-run flow child: block this request's threadpool
    thread until the run reaches a terminal status in the DB, then build the
    same response shape the old in-process path returned.

    The DB is the only child→backend channel. `prev_completed_at` is the
    transition witness for resume: the pre-resume failure timestamp, so a
    stale 'failed' row read before the child starts isn't mistaken for the
    resume's outcome. Safety nets: a child that exits without writing a
    terminal status is marked failed after a short grace; a run exceeding
    FLOW_RUNNER_TIMEOUT_SECONDS is killed and marked failed.
    """
    overall_timeout = int(os.environ.get("FLOW_RUNNER_TIMEOUT_SECONDS", "3600"))
    start = time.monotonic()
    exit_grace_until = None

    def _is_terminal(ex) -> bool:
        if ex is None or ex.status not in ("completed", "failed"):
            return False
        if prev_completed_at is not None:
            return ex.completed_at is not None and ex.completed_at != prev_completed_at
        return True

    while True:
        session = db_manager.get_session()
        try:
            ex = get_execution_by_id(session, execution_id)
            if _is_terminal(ex):
                break
            if time.monotonic() - start > overall_timeout:
                child.shutdown()
                update_execution(
                    session, execution_id,
                    status="failed",
                    error_message=f"flow run exceeded {overall_timeout}s and was killed",
                    completed_at=datetime.now(),
                )
                break
            if not child.is_alive():
                if exit_grace_until is None:
                    exit_grace_until = time.monotonic() + 2.0
                elif time.monotonic() > exit_grace_until:
                    update_execution(
                        session, execution_id,
                        status="failed",
                        error_message="flow process exited before reporting a result "
                                      "(check the backend console for its traceback)",
                        completed_at=datetime.now(),
                    )
                    break
        finally:
            session.close()
        time.sleep(0.3)

    session = db_manager.get_session()
    try:
        ex = get_execution_by_id(session, execution_id)
        status = ex.status if ex else "failed"
        result = {
            "flow_id": flow_id,
            "execution_id": execution_id,
            "status": status,
            "final_output": ex.children[-1].output_data if (status == "completed" and ex.children) else None,
        }
        if status == "failed":
            result["error"] = ex.error_message if ex else "execution record missing"
            if child.is_alive():
                live_run_store.retain(child)
        else:
            try:
                child.popen.wait(timeout=15)
            except Exception:
                pass
        return result
    finally:
        session.close()


@app.post("/flows/{flow_id}/execute")
def execute_flow_endpoint(flow_id: int, request: FlowExecuteRequest, db: Session = Depends(get_db)):
    """Execute a flow.

    LOCAL: spawns a per-run child process in the flow's environment and blocks
    until it finishes (same synchronous response as before). PRODUCTION:
    dispatches to Celery, returns 202 with execution_id for polling.
    """
    try:
        # Persist conda_env on the flow if provided (same in both modes)
        if request.conda_env:
            update_flow(db, flow_id, conda_env=request.conda_env)

        if is_hosted():
            flow = get_flow_by_id(db, flow_id)
            if not flow:
                raise HTTPException(status_code=404, detail=f"Flow {flow_id} not found")

            execution = create_execution(
                db,
                user_id=request.user_id,
                flow_id=flow_id,
                execution_type='flow',
                name=flow.name,
                input_data=request.initial_input,
                status='pending',
                started_at=datetime.now(),
            )

            # Import lazily so local mode doesn't require celery/redis installed
            from src.tasks.tasks import execute_flow_task
            execute_flow_task.delay(
                flow_id,
                request.user_id,
                request.initial_input,
                request.conda_env,
                execution.id,
                request.agent_llms,
            )

            return {"execution_id": execution.id, "status": "pending"}

        # Local: per-run child process, synchronous facade over DB polling.
        flow = get_flow_by_id(db, flow_id)
        if not flow:
            raise HTTPException(status_code=404, detail=f"Flow {flow_id} not found")

        execution = create_execution(
            db,
            user_id=request.user_id,
            flow_id=flow_id,
            execution_type='flow',
            name=flow.name,
            input_data=request.initial_input,
            status='pending',
            started_at=datetime.now(),
        )
        try:
            child = spawn_local_flow_child(
                flow_id,
                request.user_id,
                execution.id,
                request.initial_input,
                flow.conda_env,
                request.agent_llms,
            )
        except Exception as e:
            update_execution(
                db, execution.id,
                status="failed",
                error_message=f"failed to start flow process: {e}",
                completed_at=datetime.now(),
            )
            raise HTTPException(status_code=500, detail=f"Flow execution failed to start: {e}")

        return _watch_local_flow_child(child, execution.id, flow_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flow execution failed: {str(e)}")

class ExecutionResumeRequest(BaseModel):
    user_id: int = 1


@app.post("/executions/{execution_id}/resume")
def resume_execution_endpoint(execution_id: int, request: ExecutionResumeRequest, db: Session = Depends(get_db)):
    """Resume a failed flow run from its in-memory checkpoint.

    LOCAL: signals the resident flow child over stdin and blocks until the DB
    shows the outcome. HOSTED: publishes on redis 'resume:<execution_id>' for
    the resident flow-runner container. 410 means the checkpoint is gone
    (server restarted, superseded by a newer failure, or runner timed out).
    """
    ex = get_execution_by_id(db, execution_id)
    if not ex:
        raise HTTPException(status_code=404, detail="Execution not found")
    if ex.status != "failed":
        raise HTTPException(status_code=409, detail=f"Execution is '{ex.status}', not 'failed'")

    if is_hosted():
        import redis  # lazy, like the celery import in the execute endpoint
        client = redis.Redis.from_url(os.environ["CELERY_BROKER_URL"])
        try:
            receivers = client.publish(f"resume:{execution_id}", "resume")
        finally:
            client.close()
        if receivers == 0:
            raise HTTPException(status_code=410, detail="Run is no longer resumable (runner has shut down). Re-run the flow.")
        # Flip to running here so pollers never see the stale 'failed' state
        # while the container processes the resume message.
        update_execution(db, execution_id, status="running", error_message=None, completed_at=None)
        return JSONResponse(status_code=202, content={"execution_id": execution_id, "status": "running"})

    child = live_run_store.pop(execution_id)
    if child is None:
        raise HTTPException(status_code=410, detail="Run is no longer resumable (server restarted or checkpoint superseded). Re-run the flow.")
    prev_completed_at = ex.completed_at
    if not child.signal({"action": "resume"}):
        child.shutdown()
        raise HTTPException(status_code=410, detail="Run is no longer resumable (runner process has exited). Re-run the flow.")
    return _watch_local_flow_child(child, execution_id, ex.flow_id, prev_completed_at=prev_completed_at)


class ExecutionTestRequest(BaseModel):
    node_id: str
    user_id: int = 1


@app.post("/executions/{execution_id}/test")
def test_execution_node_endpoint(execution_id: int, request: ExecutionTestRequest, db: Session = Depends(get_db)):
    """Test one tool node inside a resident failed run, against its real
    ctx.state inputs.

    Transient: the run's resident process executes the node's (re-fetched, so
    freshly edited) tool and writes the outcome to the root execution's
    execution_metadata["last_test_result"] — no execution rows are created.
    This endpoint signals the resident process (stdin locally, redis hosted)
    and polls that metadata field for the matching request_id. 410 means no
    resident process holds this run anymore.
    """
    ex = get_execution_by_id(db, execution_id)
    if not ex:
        raise HTTPException(status_code=404, detail="Execution not found")
    if ex.status != "failed":
        raise HTTPException(status_code=409, detail=f"Execution is '{ex.status}', not 'failed' — only a resident failed run is testable")

    request_id = uuid.uuid4().hex
    message = {"action": "test", "node_id": request.node_id, "request_id": request_id}

    if is_hosted():
        import redis  # lazy, like the celery import in the execute endpoint
        client = redis.Redis.from_url(os.environ["CELERY_BROKER_URL"])
        try:
            receivers = client.publish(
                f"test:{execution_id}",
                json.dumps({"node_id": request.node_id, "request_id": request_id}),
            )
        finally:
            client.close()
        if receivers == 0:
            raise HTTPException(status_code=410, detail="Run is no longer resident (runner has shut down). Re-run the flow.")
    else:
        child = live_run_store.get(execution_id)
        if child is None or not child.is_alive() or not child.signal(message):
            raise HTTPException(status_code=410, detail="Run is no longer resident (server restarted or checkpoint superseded). Re-run the flow.")

    # Aligned with the test's own execution timeout so a slow-but-valid test
    # isn't abandoned; the margin covers signal latency and the DB write.
    tool_timeout = int(os.environ.get("TOOL_TIMEOUT_SECONDS", "300"))
    deadline = time.monotonic() + tool_timeout + 30
    while time.monotonic() < deadline:
        session = db_manager.get_session()
        try:
            row = get_execution_by_id(session, execution_id)
            result = (row.execution_metadata or {}).get("last_test_result") if row else None
            if result and result.get("request_id") == request_id:
                return result
        finally:
            session.close()
        time.sleep(0.3)

    return {
        "request_id": request_id,
        "node_id": request.node_id,
        "status": "error",
        "error": "test did not report a result in time",
        "error_type": "TimeoutError",
    }


@app.patch("/flows/{flow_id}", response_model=FlowResponse)
def update_flow_endpoint(flow_id: int, flow_update: FlowUpdate, db: Session = Depends(get_db)):
    """Update a flow's properties (name, description, graph_config, etc.)"""
    # Get the existing flow
    flow = get_flow_by_id(db, flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")

    # Only update fields that were provided
    update_data = flow_update.model_dump(exclude_unset=True)
    if not update_data:
        return flow

    # Update the flow
    updated_flow = update_flow(db, flow_id, **update_data)
    if not updated_flow:
        raise HTTPException(status_code=500, detail="Failed to update flow")

    return updated_flow

@app.delete("/flows/{flow_id}")
def delete_flow_endpoint(flow_id: int, db: Session = Depends(get_db)):
    """Delete a flow"""
    try:
        success = delete_flow(db, flow_id)
        if not success:
            raise HTTPException(status_code=404, detail="Flow not found")
        return {"status": "success", "message": f"Flow {flow_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flow deletion failed: {str(e)}")

# ─── Execution Endpoints ───

def _execution_to_tree(execution) -> dict:
    """Recursively convert an Execution ORM object to a dict tree."""
    return {
        "id": execution.id,
        "parent_id": execution.parent_id,
        "execution_type": execution.execution_type,
        "node_id": execution.node_id,
        "name": execution.name,
        "sequence": execution.sequence,
        "input_data": execution.input_data,
        "output_data": execution.output_data,
        "status": execution.status,
        "error_message": execution.error_message,
        "started_at": execution.started_at,
        "completed_at": execution.completed_at,
        "execution_metadata": execution.execution_metadata,
        "langfuse_trace_id": execution.langfuse_trace_id,
        "children": [_execution_to_tree(child) for child in execution.children]
    }

@app.get("/executions", response_model=List[ExecutionListItem])
def list_executions_endpoint(user_id: int, limit: int = 50, offset: int = 0, db: Session = Depends(get_db)):
    """List top-level executions for a user, newest first."""
    executions = get_user_executions(db, user_id, limit=limit, offset=offset)
    return executions

@app.get("/executions/{execution_id}")
def get_execution_endpoint(execution_id: int, db: Session = Depends(get_db)):
    """Get a full execution tree by ID (recursively includes all children)."""
    execution = get_execution_by_id(db, execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    return _execution_to_tree(execution)

@app.get("/executions/{execution_id}/trace")
def get_execution_trace_endpoint(execution_id: int, db: Session = Depends(get_db)):
    """Fetch the LangFuse trace for an execution's agent call."""
    execution = get_execution_by_id(db, execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    if not execution.langfuse_trace_id:
        raise HTTPException(status_code=404, detail="No LangFuse trace for this execution")

    try:
        from src.observability.langfuse_tracing import langfuse_client, LANGFUSE_AVAILABLE
        if not LANGFUSE_AVAILABLE or not langfuse_client:
            raise HTTPException(status_code=503, detail="LangFuse not configured")

        trace = langfuse_client.api.trace.get(execution.langfuse_trace_id)

        # Convert observations to serializable dicts
        observations = []
        for obs in (trace.observations or []):
            observations.append({
                "id": obs.id,
                "name": obs.name,
                "type": obs.type,
                "input": obs.input,
                "output": obs.output,
                "model": getattr(obs, 'model', None),
                "start_time": str(obs.start_time) if obs.start_time else None,
                "end_time": str(obs.end_time) if obs.end_time else None,
                "usage": {
                    "input": obs.usage.input if obs.usage else None,
                    "output": obs.usage.output if obs.usage else None,
                    "total": obs.usage.total if obs.usage else None,
                } if obs.usage else None,
                "level": getattr(obs, 'level', None),
                "status_message": getattr(obs, 'status_message', None),
            })

        return {
            "trace_id": trace.id,
            "name": trace.name,
            "input": trace.input,
            "output": trace.output,
            "observations": observations,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch LangFuse trace: {str(e)}")

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

# LLM Provider Config Endpoints
@app.get("/llm-providers/config", response_model=LLMProvidersConfigResponse)
def get_llm_providers_config(user_id: Optional[int] = Query(None)):
    """Load LLM provider configuration with masked credentials.

    In hosted mode, loads per-user config from the database.
    In local mode, loads from ~/.llm_hub/config.yaml (user_id ignored).
    """
    try:
        from src.utils import mask_credentials

        config = load_request_llm_config(user_id)

        # Mask sensitive credentials before sending to frontend
        config['models'] = mask_credentials(config.get('models', []))

        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load LLM config: {str(e)}")

@app.post("/llm-providers/config")
def save_llm_providers_config(config: LLMProvidersConfigRequest, user_id: Optional[int] = Query(None)):
    """Save LLM provider configuration.

    In hosted mode, saves per-user config to the database.
    In local mode, saves to ~/.llm_hub/config.yaml (user_id ignored).
    """
    try:
        from src.utils import restore_masked_credentials

        # Convert Pydantic models to dictionaries
        models_list = [model.model_dump() for model in config.models]

        # Load existing config to restore any masked credentials
        existing_config = load_request_llm_config(user_id)
        existing_models = existing_config.get('models', [])

        # Restore masked credentials (if any)
        models_list = restore_masked_credentials(models_list, existing_models)

        # Save config: DB in HOSTED, YAML in LOCAL
        if is_hosted() and user_id is not None:
            session = db_manager.get_session()
            try:
                sync_user_llm_configs(session, user_id, models_list)
            finally:
                session.close()
        else:
            save_llm_provider_config(models_list)

        return {
            "status": "success",
            "message": "LLM provider configuration saved successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save LLM config: {str(e)}")


# --- Evaluation Endpoints ---

class EvalPromptGenerateRequest(BaseModel):
    eval_name: str
    eval_description: str = ""
    score_type: str
    score_categories: Optional[List[Dict[str, Any]]] = None
    return_fields: Optional[List[str]] = None
    input_variables: List[str] = ["output"]
    model: str = ""
    additional_instructions: Optional[str] = None
    user_id: Optional[int] = None

@app.post("/evaluations/generate-prompt")
async def generate_eval_prompt_endpoint(request: EvalPromptGenerateRequest, db: Session = Depends(get_db)):
    """Generate a judge system prompt for an evaluation using AI (streaming)."""
    from src.ai_integrations.generate_eval_prompt import generate_eval_prompt_stream

    if not request.model:
        raise HTTPException(status_code=400, detail="LLM model is required")

    llm_config = _resolve_llm_config(request.model, request.user_id)

    async def stream():
        async for chunk in generate_eval_prompt_stream(
            session=db,
            eval_name=request.eval_name,
            eval_description=request.eval_description,
            score_type=request.score_type,
            score_categories=request.score_categories,
            return_fields=request.return_fields,
            input_variables=request.input_variables,
            llm_config=llm_config,
            additional_instructions=request.additional_instructions,
        ):
            yield chunk

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/evaluations/", response_model=EvaluationResponse)
def create_evaluation_endpoint(data: EvaluationCreate, user_id: int, db: Session = Depends(get_db)):
    try:
        evaluation = create_evaluation(
            db,
            user_id=user_id,
            name=data.name,
            description=data.description,
            judge_system_prompt=data.judge_system_prompt,
            judge_user_prompt=data.judge_user_prompt,
            scoring_rubric=data.scoring_rubric,
            score_type=data.score_type,
            score_categories=data.score_categories,
            llm_provider=data.llm_provider,
            input_variables=data.input_variables,
            return_fields=data.return_fields,
            is_public=data.is_public,
        )
        return evaluation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create evaluation: {str(e)}")

@app.get("/evaluations/", response_model=List[EvaluationResponse])
def list_evaluations_endpoint(user_id: int, db: Session = Depends(get_db)):
    return get_evaluations_by_user(db, user_id)

@app.get("/evaluations/{evaluation_id}", response_model=EvaluationResponse)
def get_evaluation_endpoint(evaluation_id: int, db: Session = Depends(get_db)):
    evaluation = get_evaluation_by_id(db, evaluation_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return evaluation

@app.patch("/evaluations/{evaluation_id}", response_model=EvaluationResponse)
def update_evaluation_endpoint(evaluation_id: int, data: EvaluationUpdate, db: Session = Depends(get_db)):
    update_data = data.model_dump(exclude_unset=True)
    evaluation = update_evaluation(db, evaluation_id, **update_data)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return evaluation

@app.delete("/evaluations/{evaluation_id}")
def delete_evaluation_endpoint(evaluation_id: int, db: Session = Depends(get_db)):
    if not delete_evaluation(db, evaluation_id):
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return {"status": "deleted"}

@app.post("/executions/{execution_id}/evaluate", response_model=List[EvaluationResultResponse])
async def evaluate_execution_endpoint(execution_id: int, data: EvaluateRequest, db: Session = Depends(get_db)):
    """Run one or more evaluations against an execution."""
    execution = get_execution_by_id(db, execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    if not execution.langfuse_trace_id:
        raise HTTPException(status_code=400, detail="Execution has no LangFuse trace — cannot evaluate")

    from src.executors.evaluation_executor import EvaluationExecutor
    llm_config = load_request_llm_config(data.user_id, db)
    executor = EvaluationExecutor(db, llm_config=llm_config)

    results = []
    for eval_id in data.evaluation_ids:
        result = await executor.evaluate(
            evaluation_id=eval_id,
            execution_id=execution_id,
            user_id=data.user_id,
            llm_provider=data.llm_provider,
        )
        # Reload the result from DB to get full object for response
        from src.database.database import get_evaluation_results_by_execution
        all_results = get_evaluation_results_by_execution(db, execution_id)
        eval_result = next((r for r in all_results if r.id == result["id"]), None)
        if eval_result:
            eval_obj = get_evaluation_by_id(db, eval_result.evaluation_id)
            results.append(EvaluationResultResponse(
                id=eval_result.id,
                evaluation_id=eval_result.evaluation_id,
                evaluation_name=eval_obj.name if eval_obj else None,
                execution_id=eval_result.execution_id,
                langfuse_trace_id=eval_result.langfuse_trace_id,
                langfuse_score_id=eval_result.langfuse_score_id,
                status=eval_result.status,
                error_message=eval_result.error_message,
                created_at=eval_result.created_at,
                completed_at=eval_result.completed_at,
            ))

    return results

@app.get("/executions/{execution_id}/evaluations", response_model=List[EvaluationResultResponse])
def get_execution_evaluations_endpoint(execution_id: int, db: Session = Depends(get_db)):
    """Get all evaluation results for an execution."""
    results = get_evaluation_results_by_execution(db, execution_id)
    response = []
    for r in results:
        eval_obj = get_evaluation_by_id(db, r.evaluation_id)
        response.append(EvaluationResultResponse(
            id=r.id,
            evaluation_id=r.evaluation_id,
            evaluation_name=eval_obj.name if eval_obj else None,
            execution_id=r.execution_id,
            langfuse_trace_id=r.langfuse_trace_id,
            langfuse_score_id=r.langfuse_score_id,
            status=r.status,
            error_message=r.error_message,
            created_at=r.created_at,
            completed_at=r.completed_at,
        ))
    return response

@app.get("/executions/{execution_id}/scores")
def get_execution_scores_endpoint(execution_id: int, db: Session = Depends(get_db)):
    """Fetch actual score data from LangFuse for an execution's trace."""
    execution = get_execution_by_id(db, execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    if not execution.langfuse_trace_id:
        raise HTTPException(status_code=404, detail="No LangFuse trace for this execution")

    try:
        from src.observability.langfuse_tracing import langfuse_client, LANGFUSE_AVAILABLE
        if not LANGFUSE_AVAILABLE or not langfuse_client:
            raise HTTPException(status_code=503, detail="LangFuse not configured")

        trace = langfuse_client.api.trace.get(execution.langfuse_trace_id)

        scores = []
        for s in (trace.scores or []):
            scores.append({
                "id": s.id,
                "name": s.name,
                "value": s.value,
                "comment": getattr(s, "comment", None),
                "data_type": getattr(s, "data_type", None),
                "created_at": str(s.created_at) if hasattr(s, "created_at") and s.created_at else None,
            })

        return {"trace_id": execution.langfuse_trace_id, "scores": scores}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch LangFuse scores: {str(e)}")