#!/usr/bin/env python3
"""
Factory for creating Agent and Tool instances from database configurations.
"""

import importlib
import inspect
from typing import Dict, Any, List, Optional
from tools.tool_and_agent_creator import create_tool, CustomAgent
from llm_setup import create_llm
from database.database_setup import DatabaseManager, Tool
import logging

logger = logging.getLogger(__name__)

class Factory:
    """Factory for creating Agent and Tool instances from database configurations."""
    
    def __init__(self):

        self.db_manager = DatabaseManager()

    
    
    def create_tool_from_database(self, tool_id: int):
        """
        Create a tool instance from a database Tool by ID.
        
        Args:
            tool_id: ID of the tool in the database
            
        Returns:
            Tool instance or None if creation fails
        """
        try:
            session = self.db_manager.get_session()
            
            try:
                # Get tool from database
                db_tool = session.query(Tool).filter(Tool.id == tool_id).first()
                
                if not db_tool:
                    logger.error(f"Tool with ID {tool_id} not found")
                    return None
                
                if db_tool.tool_type == "function":
                    # For function-based tools, get the function from registry
                    if db_tool.function_name and db_tool.function_name in self.function_registry:
                        func = self.function_registry[db_tool.function_name]
                        
                        # Create tool using create_tool function
                        tool = create_tool(
                            name=db_tool.name,
                            description=db_tool.description,
                            func=func
                        )
                        return tool
                    else:
                        logger.error(f"Function {db_tool.function_name} not found in registry for tool {db_tool.name}")
                        return None

                elif db_tool.tool_type == "custom":
                    # For custom tools, you might need to execute the function_code
                    # This would require careful security considerations
                    logger.warning(f"Custom tools not yet implemented for tool {db_tool.name}")
                    return None
                
                else:
                    logger.error(f"Unknown tool type: {db_tool.tool_type} for tool {db_tool.name}")
                    return None
                    
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error creating tool with ID {tool_id}: {str(e)}")
            return None
    
    def create_agent_from_config(self, agent_config: Dict[str, Any]) -> CustomAgent:
        """
        Create a CustomAgent instance from a database configuration.
        
        Args:
            agent_config: Dictionary containing agent configuration from database
            
        Returns:
            CustomAgent instance
        """
        try:
            # Extract configuration
            name = agent_config["name"]
            description = agent_config["description"]
            llm_config = agent_config["llm_config"]
            tools = agent_config["tools"]
            agent_metadata = agent_config.get("agent_metadata", {})
            
            # Create LLM instance
            llm_provider = llm_config["provider"]
            llm_model = llm_config["model"]
            llm_temperature = llm_config["temperature"]
            llm = create_llm(provider=llm_provider, model=llm_model, temperature=llm_temperature)
            
            # Create tools from configuration
            tools = []
            for tool_id in tools:
                tool = self.create_tool_from_database(tool_id)
                if tool:
                    tools.append(tool)
                else:
                    logger.warning(f"Failed to create tool with ID {tool_id}")           

                tools.append(tool)
            
            # Create CustomAgent instance
            custom_agent = CustomAgent(
                name=name,
                description=description,
                llm=llm,
                tools=tools,
                **agent_metadata
            )
            
            return custom_agent
            
        except Exception as e:
            logger.error(f"Error creating agent from config: {str(e)}")
            raise
    
    def create_agent_from_database(self, db_agent) -> CustomAgent:
        """
        Create a CustomAgent instance from a database Agent object.
        
        Args:
            db_agent: Agent object from database
            
        Returns:
            CustomAgent instance
        """
        try:
            
            # Get tools from the database relationship
            tools = []
            session = self.db_manager.get_session()
            
            try:
                # Refresh the agent to ensure relationships are loaded
                session.refresh(db_agent)
                
                # Get tools associated with this agent
                for db_tool in db_agent.tools:
                    tool = self.create_tool_from_database(db_tool.id)
                    if tool:
                        tools.append(tool)
                    else:
                        logger.warning(f"Failed to create tool {db_tool.name} for agent {db_agent.name}")
                
            finally:
                session.close()

            agent_config = {
                'name': db_agent.name,
                'description': db_agent.description,
                'llm_config': db_agent.llm_config,
                'tools': db_agent.tools,
                'agent_metadata': db_agent.agent_metadata if db_agent.agent_metadata else {}
            }
            
            custom_agent = self.create_agent_from_config(agent_config)
                    
            return custom_agent
            
        except Exception as e:
            logger.error(f"Error creating agent from database: {str(e)}")
            raise
    
    def create_agent_by_id(self, agent_id: int) -> Optional[CustomAgent]:
        """
        Create a CustomAgent instance by agent ID.
        
        Args:
            agent_id: ID of the agent in the database
            
        Returns:
            CustomAgent instance or None if not found
        """
        try:
            session = self.db_manager.get_session()
            
            try:
                # Get agent from database
                from database.database_setup import Agent
                db_agent = session.query(Agent).filter(Agent.id == agent_id).first()
                
                if not db_agent:
                    logger.error(f"Agent with ID {agent_id} not found")
                    return None
                
                return self.create_agent_from_database(db_agent)
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error creating agent by ID {agent_id}: {str(e)}")
            return None
    
    def get_agent_tools(self, agent_id: int) -> List[Tool]:
        """
        Get all tools associated with an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of Tool objects
        """
        try:
            session = self.db_manager.get_session()
            
            try:
                from database.database_setup import Agent
                db_agent = session.query(Agent).filter(Agent.id == agent_id).first()
                
                if not db_agent:
                    logger.error(f"Agent with ID {agent_id} not found")
                    return []
                
                # Refresh to ensure relationships are loaded
                session.refresh(db_agent)
                return list(db_agent.tools)
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error getting tools for agent {agent_id}: {str(e)}")
            return []
    
    def validate_agent_config(self, agent_config: Dict[str, Any]) -> List[str]:
        """
        Validate an agent configuration and return any errors.
        
        Args:
            agent_config: Dictionary containing agent configuration
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Check required fields
        required_fields = ["name", "description", "llm_config", "tools_config"]
        for field in required_fields:
            if field not in agent_config:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return errors
        
        # Validate LLM config
        try:
            create_llm(agent_config["llm_config"])
        except Exception as e:
            errors.append(f"Invalid LLM config: {str(e)}")
        
        
        return errors

# Global factory instance
factory = Factory() 