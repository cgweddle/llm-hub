"""
Agent Factory                                                                                                 
Creates LangGraph/LangChain agents from database tools and saves them back to the database
""" 

import json 
import logging 
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
from langchain_litellm import ChatLiteLLM

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database.database import create_agent as db_create_agent, get_session, get_available_tools, get_tool_by_id
from llm_setup import create_llm

logger = logging.getLogger(__name__)

class AgentConfig(BaseModel):
    """Configuration for creating an agent with validation"""
    name: str = Field(..., min_length=1, description="Agent Name")
    description: str = Field(..., min_length=1, description="Agent Description")
    llm: ChatLiteLLM = Field(..., description="LLM")
    agent_description: str = Field(..., min_length=1, description="Agent Description")
    tool_ids: Optional[List[int]] = Field(default=None, description="List of tool IDs in the Agent")
    user_id: int = Field(..., description="ID of User Creator")
    is_public: bool = Field(default=False, description="Whether the Agent is public")

class DatabaseToolConvertor:
    """Convert tools in the database to LangChain tools"""

class LanggraphAgentFactory:
    def __init__(self, config: AgentConfig):
        self.transformer = DatabaseToLangGraphTransformer()
        self.tools = []
        for tool_id in config.tool_ids:
            self.tools.append(self.transformer.transform_tool(tool_id))
    def _generate_system_prompt(self, config: AgentConfig):
        agent_description = config.agent_description
        tool_descriptions = {}
        for tool in self.tools:
            tool_name = tool.name
            tool_description = tool.description
            tool_descriptions[tool_name] = tool_description
        tools_str = "\n".join([f"{tool} tool: {tool_descriptions[tool]}" for tool in tool_descriptions.keys()])
        agent_prompt = f"You are an AI agent tasked with {agent_description}. "
        agent_prompt += f"You have access to the following tools: \n{tool_descriptions}\n"
        agent_prompt += ("Do not ask for clarification. "
                        "Respond only with the result of the tool. Do not add additional text or information. "
                        "Only address the parts of the prompt that fall within your capabilities. "
                        "If a task is outside your description or the scope of your tools, do not attempt to answer it. "
                        "Simply end and allow a different agent to handle that part of the request. "
                        "Do not add extra text or information about why you are ending.")
        return agent_prompt

    def create_agent(self, config: AgentConfig):
        

    