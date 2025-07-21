# api.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.database.database import get_session, create_agent, get_user_agents, create_user
from scripts.database.database_setup import DatabaseManager
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime
from passlib.hash import bcrypt

app = FastAPI()
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

@app.post("/agents/", response_model=AgentResponse)
def create_agent(agent_data: AgentCreate, user_id: int):
    """Create a new agent"""
    db_session = get_session()
    agent = create_agent(
        session = db_session,
        user_id=user_id,
        name=agent_data.name,
        description=agent_data.description,
        agent_type=agent_data.agent_type,
        system_prompt=agent_data.system_prompt,
        llm_config=agent_data.llm_config,
        tools=agent_data.tools,
        agent_metadata=agent_data.agent_metadata
    )
    return agent

@app.get("/agents/", response_model=List[AgentResponse])
def get_user_agents(user_id: int):
    """Get all agents for a user"""
    db_session = get_session()
    return get_user_agents(session = db_session, user_id = user_id)

@app.post("/agents/{agent_id}/execute")
def execute_agent(agent_id: int, user_id: int, input_data: str, db: Session = Depends(get_db)):
    """Execute an agent"""
    agent_service = AgentService(db)
    # You'll need to get the LLM instance here
    llm = get_llm_instance()  # Implement this based on your LLM setup
    result = agent_service.execute_agent(user_id, agent_id, input_data, llm)
    return {"result": result}

@app.post("/users/", response_model=UserResponse)
def create_user_endpoint(user_data: UserCreate):
    """Create a new user"""
    print('trying database connectoin')
    db_session = get_session()
    print('database connected')
    # Hash the password
    password_hash = bcrypt.hash(user_data.password)
    user = create_user(
        session=db_session,
        username=user_data.username,
        email=user_data.email,
        password_hash=password_hash
    )
    return user