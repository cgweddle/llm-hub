# api.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
    get_tool_by_id, update_tool, delete_tool, get_flow_by_id, update_flow, delete_flow,
    get_agent_by_id, update_agent, delete_agent,
    get_execution_by_id, get_user_executions
)
from src.utils import load_llm_provider_config, save_llm_provider_config
from src.database.database_setup import DatabaseManager
from src.validate.tool_compatibility import validate_two_tools, validate_tool_compatibility, validate_connection
from src.executors.flow_executor import FlowExecutor
from src.executors.agent_executor import AgentExecutor
from src.factories.python_script_tool_factory import PythonScriptToolFactory
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
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

class AgentExecuteRequest(BaseModel):
    user_id: int
    input_data: str
    stream: bool = False

class AgentExecuteResponse(BaseModel):
    execution_id: int
    status: str
    result: Any
    messages: List[Dict[str, Any]]
    cost: Optional[Dict[str, Any]] = None

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

class UserPromptGenerateRequest(BaseModel):
    agent_name: str
    agent_description: str
    tool_names: List[str] = []
    model: str
    generated_system_prompt: str
    additional_instructions: Optional[str] = None


class CodeGenerateRequest(BaseModel):
    tool_name: str
    tool_description: str
    provider: str  # e.g., 'anthropic', 'openai', 'gemini', 'lmstudio'
    model: str     # e.g., 'claude-3-5-sonnet-20241022'
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    additional_instructions: Optional[str] = None

class CodeGenerateResponse(BaseModel):
    script_code: str
    main_function: str

class CodeEditRequest(BaseModel):
    existing_code: str
    editing_instructions: str
    tool_name: str
    tool_description: str
    provider: str  # e.g., 'anthropic', 'openai', 'gemini', 'lmstudio'
    model: str     # e.g., 'claude-3-5-sonnet-20241022'
    api_key: Optional[str] = None
    base_url: Optional[str] = None

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

@app.post("/agents/{agent_id}/execute", response_model=AgentExecuteResponse)
async def execute_agent_endpoint(
    agent_id: int,
    request: AgentExecuteRequest,
    db: Session = Depends(get_db)
):
    """
    Execute an agent (supports both Google ADK ReAct and PydanticAI agents).

    This endpoint automatically detects the agent type and routes to the appropriate
    executor. Supports both standard and streaming execution modes.

    Args:
        agent_id: ID of the agent to execute
        request: AgentExecuteRequest with user_id, input_data, and optional stream flag
        db: Database session (injected)

    Returns:
        AgentExecuteResponse with execution results

    Example:
        POST /agents/5/execute
        {
            "user_id": 1,
            "input_data": "What is 2+2?",
            "stream": false
        }
    """
    try:
        executor = AgentExecutor(db)

        # Handle streaming requests
        if request.stream:
            async def stream_generator():
                try:
                    result = await executor.execute_agent(
                        agent_id=agent_id,
                        user_id=request.user_id,
                        input_data=request.input_data,
                        stream=True
                    )

                    # result is an async generator for streaming
                    if hasattr(result, '__aiter__'):
                        async for chunk in result:
                            yield f"data: {json.dumps(chunk)}\n\n"
                    else:
                        # Fallback if not streaming (shouldn't happen but handle it)
                        yield f"data: {json.dumps(result)}\n\n"

                except Exception as e:
                    error_chunk = {
                        "type": "error",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )

        # Standard (non-streaming) execution
        else:
            result = await executor.execute_agent(
                agent_id=agent_id,
                user_id=request.user_id,
                input_data=request.input_data,
                stream=False
            )
            return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

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
def generate_system_prompt_endpoint(request: SystemPromptGenerateRequest, db: Session = Depends(get_db)):
    """Generate a system prompt for an agent using AI with streaming"""
    try:
        from src.ai_integrations.generate_agent_system_prompt import generate_system_prompt_stream

        def stream_generator():
            try:
                for chunk in generate_system_prompt_stream(
                    session=db,
                    agent_name=request.agent_name,
                    agent_description=request.agent_description,
                    tool_names=request.tool_names,
                    llm_model=request.model,
                    additional_instructions=request.additional_instructions
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
def generate_user_prompt_endpoint(request: UserPromptGenerateRequest, db: Session = Depends(get_db)):
    """Generate a task-specific user prompt for an agent using AI with streaming"""
    try:
        from src.ai_integrations.generate_agent_system_prompt import generate_user_prompt_stream

        def stream_generator():
            try:
                for chunk in generate_user_prompt_stream(
                    session=db,
                    agent_name=request.agent_name,
                    agent_description=request.agent_description,
                    tool_names=request.tool_names,
                    llm_model=request.model,
                    generated_system_prompt=request.generated_system_prompt,
                    additional_instructions=request.additional_instructions
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
            user_id=user_id
        )
        
        # Fetch the created tool
        created_tool = get_tool_by_id(db, tool_id)
        if not created_tool:
            raise HTTPException(status_code=500, detail="Tool was created but could not be retrieved")
        
        # Update is_public if needed (factory doesn't handle this)
        if created_tool.is_public != tool_data.is_public:
            update_tool(db, tool_id, is_public=tool_data.is_public)
            created_tool = get_tool_by_id(db, tool_id)
        
        return created_tool
        
    except ValueError as e:
        # Factory validation errors (e.g., invalid syntax, missing function)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Other errors
        raise HTTPException(status_code=500, detail=f"Failed to create python script tool: {str(e)}")

@app.post("/tools/generate-code")
def generate_tool_code_endpoint(request: CodeGenerateRequest, db: Session = Depends(get_db)):
    """Generate Python tool code using AI with user-selected LLM (streaming)"""
    try:
        from src.ai_integrations.generate_python_tools import generate_tool_code_stream

        def stream_generator():
            try:
                for chunk in generate_tool_code_stream(
                    session=db,
                    tool_name=request.tool_name,
                    tool_description=request.tool_description,
                    provider=request.provider,
                    model=request.model,
                    api_key=request.api_key,
                    base_url=request.base_url,
                    additional_instructions=request.additional_instructions
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
def edit_tool_code_endpoint(request: CodeEditRequest, db: Session = Depends(get_db)):
    """Edit existing Python tool code using AI with user-selected LLM (streaming)"""
    try:
        from src.ai_integrations.generate_python_tools import edit_tool_code_stream

        def stream_generator():
            try:
                for chunk in edit_tool_code_stream(
                    session=db,
                    existing_code=request.existing_code,
                    editing_instructions=request.editing_instructions,
                    tool_name=request.tool_name,
                    tool_description=request.tool_description,
                    provider=request.provider,
                    model=request.model,
                    api_key=request.api_key,
                    base_url=request.base_url
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

@app.post("/flows/{flow_id}/execute")
def execute_flow_endpoint(flow_id: int, request: FlowExecuteRequest, db: Session = Depends(get_db)):
    """Execute a flow"""
    try:
        executor = FlowExecutor(db, flow_id, user_id=request.user_id)
        result = executor.execute_flow(request.initial_input, request.conda_env)

        # Update the flow's conda_env if provided
        if request.conda_env:
            update_flow(db, flow_id, conda_env=request.conda_env)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flow execution failed: {str(e)}")

@app.post("/flows/{flow_id}/resume")
def resume_flow_endpoint(flow_id: int, execution_trace: list, resume_input: Optional[dict] = None, user_id: int = 1, db: Session = Depends(get_db)):
    """Resume a failed flow"""
    try:
        executor = FlowExecutor(db, flow_id, user_id=user_id)
        result = executor.resume_flow(flow_id, execution_trace, resume_input)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flow resume failed: {str(e)}")

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
        from src.executors.agent_executor import langfuse_client, LANGFUSE_AVAILABLE
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
def get_llm_providers_config():
    """Load LLM provider configuration from ~/.llm_hub/config.yaml with masked credentials"""
    try:
        from src.utils import mask_credentials

        config = load_llm_provider_config()

        # Mask sensitive credentials before sending to frontend
        config['models'] = mask_credentials(config.get('models', []))

        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load LLM config: {str(e)}")

@app.post("/llm-providers/config")
def save_llm_providers_config(config: LLMProvidersConfigRequest):
    """Save LLM provider configuration to ~/.llm_hub/config.yaml"""
    try:
        from src.utils import get_llm_hub_config_path, restore_masked_credentials

        # Convert Pydantic models to dictionaries
        models_list = [model.model_dump() for model in config.models]

        # Load existing config to restore any masked credentials
        existing_config = load_llm_provider_config()
        existing_models = existing_config.get('models', [])

        # Restore masked credentials (if any)
        models_list = restore_masked_credentials(models_list, existing_models)

        # Save to config file
        save_llm_provider_config(models=models_list)

        return {
            "status": "success",
            "message": "LLM provider configuration saved successfully",
            "config_path": str(get_llm_hub_config_path())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save LLM config: {str(e)}")