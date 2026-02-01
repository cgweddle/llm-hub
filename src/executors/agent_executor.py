"""
Unified Agent Executor
Handles execution for both Google ADK ReAct agents and PydanticAI agents

Features:
- Type-based routing (react vs pydanticai)
- Execution record management
- Message history storage
- Streaming support for PydanticAI
- Automatic retry with exponential backoff for transient failures
- Error handling and logging
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, AsyncGenerator, Optional
from sqlalchemy.orm import Session

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database.database import get_agent_by_id
from database.database_setup import Execution, Message
from factories.agent_factory import ReactAgentFactory
from factories.pydanticai_agent_factory import PydanticAIAgentFactory

# Import retry utilities
try:
    from utils.retry import (
        RetryConfig,
        retry_async,
        DEFAULT_LLM_RETRY_CONFIG,
    )
    RETRY_AVAILABLE = True
except ImportError:
    RETRY_AVAILABLE = False
    RetryConfig = None
    DEFAULT_LLM_RETRY_CONFIG = None

logger = logging.getLogger(__name__)


class AgentExecutor:
    """
    Unified agent execution service.

    Routes execution to the appropriate agent type (Google ADK or PydanticAI),
    manages execution lifecycle, and stores results in the database.

    Features:
    - Type-based routing (react vs pydanticai)
    - Execution record management
    - Message history storage
    - Streaming support for PydanticAI
    - Automatic retry with exponential backoff
    - Error handling and logging
    """

    def __init__(
        self,
        session: Session,
        retry_config: Optional["RetryConfig"] = None,
        enable_retry: bool = True,
    ):
        """
        Initialize the agent executor.

        Args:
            session: Database session for loading agents and storing execution results
            retry_config: Optional retry configuration. Uses DEFAULT_LLM_RETRY_CONFIG if None.
            enable_retry: Whether to enable automatic retry on transient failures (default: True)
        """
        self.session = session
        self.react_factory = ReactAgentFactory(session)
        self.pydanticai_factory = PydanticAIAgentFactory(session)

        # Configure retry behavior
        self.enable_retry = enable_retry and RETRY_AVAILABLE
        if self.enable_retry:
            self.retry_config = retry_config or DEFAULT_LLM_RETRY_CONFIG
            logger.debug(
                f"Retry enabled: max_retries={self.retry_config.max_retries}, "
                f"base_delay={self.retry_config.base_delay}s"
            )
        else:
            self.retry_config = None
            if enable_retry and not RETRY_AVAILABLE:
                logger.warning("Retry requested but retry utilities not available")

    async def execute_agent(
        self,
        agent_id: int,
        user_id: int,
        input_data: str,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Execute an agent and store results in the database.

        Args:
            agent_id: ID of the agent to execute
            user_id: ID of the user executing the agent
            input_data: User input/query for the agent
            stream: Whether to stream responses (only supported for PydanticAI)

        Returns:
            Dict with execution results:
            {
                "execution_id": int,
                "status": "completed" | "failed",
                "result": Any,
                "messages": List[Dict],
                "error": Optional[str],
                "cost": Optional[Dict] (for PydanticAI)
            }

        Raises:
            ValueError: If agent not found or invalid configuration
            RuntimeError: If execution fails

        Example:
            >>> executor = AgentExecutor(db_session)
            >>> result = await executor.execute_agent(
            ...     agent_id=5,
            ...     user_id=1,
            ...     input_data="What is 2+2?"
            ... )
        """
        # Load agent record to determine type
        agent_record = get_agent_by_id(self.session, agent_id)
        if not agent_record:
            raise ValueError(f"Agent with ID {agent_id} not found")

        logger.info(
            f"Executing agent: {agent_record.name} "
            f"(type: {agent_record.agent_type}, user: {user_id}, stream: {stream})"
        )

        # Create execution record
        execution = self._create_execution(agent_id, user_id, input_data)

        try:
            # Route to appropriate executor based on agent type
            if agent_record.agent_type == "react":
                result = await self._execute_react_agent(agent_record, input_data, execution)

            elif agent_record.agent_type == "pydanticai":
                if stream:
                    # For streaming, return async generator
                    return await self._execute_pydanticai_agent_stream(
                        agent_record, input_data, execution
                    )
                else:
                    result = await self._execute_pydanticai_agent(agent_record, input_data, execution)

            else:
                raise ValueError(f"Unknown agent type: {agent_record.agent_type}")

            # Update execution with results
            self._complete_execution(execution, result)

            logger.info(f"✓ Agent execution completed: {execution.id}")
            return result

        except Exception as e:
            # Handle execution failure
            self._fail_execution(execution, str(e))
            logger.error(f"✗ Agent execution failed: {e}")
            raise RuntimeError(f"Agent execution failed: {e}")

    async def _execute_react_agent(
        self,
        agent_record,
        input_data: str,
        execution: Execution
    ) -> Dict[str, Any]:
        """
        Execute a Google ADK ReAct agent with automatic retry on transient failures.

        Args:
            agent_record: Database Agent object
            input_data: User input
            execution: Execution record

        Returns:
            Dict with execution results
        """
        logger.debug(f"Creating ReAct agent from database: {agent_record.id}")

        # Create agent using ReactAgentFactory
        agent = self.react_factory.create_from_database(agent_record.id)

        # Define the execution function
        async def run_agent():
            return await agent.run_async(input_data)

        # Run agent with retry if enabled
        if self.enable_retry:
            def on_retry(attempt: int, exception: Exception, delay: float):
                logger.warning(
                    f"ReAct agent execution retry {attempt + 1}: {type(exception).__name__}"
                )

            result = await retry_async(run_agent, config=self.retry_config, on_retry=on_retry)
        else:
            result = await run_agent()

        # Store messages from agent history
        history = result.get("history", [])
        self._store_messages(execution.id, history, message_type="react")

        return {
            "execution_id": execution.id,
            "status": "completed",
            "result": result.get("answer", ""),
            "messages": history,
            "model": result.get("model", "unknown"),
            "agent_type": "react"
        }

    async def _execute_pydanticai_agent(
        self,
        agent_record,
        input_data: str,
        execution: Execution
    ) -> Dict[str, Any]:
        """
        Execute a PydanticAI agent (non-streaming) with automatic retry on transient failures.

        Args:
            agent_record: Database Agent object
            input_data: User input
            execution: Execution record

        Returns:
            Dict with execution results including structured output
        """
        logger.debug(f"Creating PydanticAI agent from database: {agent_record.id}")

        # Create agent using PydanticAIAgentFactory
        agent = self.pydanticai_factory.create_from_database(agent_record.id)

        # Define the execution function
        async def run_agent():
            return await agent.run(input_data)

        # Run agent with retry if enabled
        if self.enable_retry:
            def on_retry(attempt: int, exception: Exception, delay: float):
                logger.warning(
                    f"PydanticAI agent execution retry {attempt + 1}: {type(exception).__name__}"
                )

            result = await retry_async(run_agent, config=self.retry_config, on_retry=on_retry)
        else:
            result = await run_agent()

        # Store messages from PydanticAI result
        self._store_pydanticai_messages(execution.id, result)

        # Extract result data (structured output if configured)
        result_data = result.data
        if hasattr(result_data, 'model_dump'):
            # If it's a Pydantic model, serialize it
            result_data = result_data.model_dump()

        return {
            "execution_id": execution.id,
            "status": "completed",
            "result": result_data,
            "messages": self._format_pydanticai_messages(result),
            "cost": self._extract_cost(result),
            "model": agent_record.llm_config.get("model", "unknown"),
            "agent_type": "pydanticai"
        }

    async def _execute_pydanticai_agent_stream(
        self,
        agent_record,
        input_data: str,
        execution: Execution
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute a PydanticAI agent with streaming responses.

        Args:
            agent_record: Database Agent object
            input_data: User input
            execution: Execution record

        Yields:
            Dict chunks with streaming data:
            - {"type": "message", "content": str, "timestamp": str}
            - {"type": "complete", "result": Any, "cost": Dict}
        """
        logger.debug(f"Creating PydanticAI agent with streaming: {agent_record.id}")

        # Create agent using PydanticAIAgentFactory
        agent = self.pydanticai_factory.create_from_database(agent_record.id)

        # Stream agent execution
        async with agent.run_stream(input_data) as stream:
            # Stream messages as they arrive
            async for message in stream:
                yield {
                    "type": "message",
                    "content": str(message.content) if hasattr(message, 'content') else str(message),
                    "timestamp": datetime.now().isoformat(),
                    "execution_id": execution.id
                }

            # Get final result after streaming completes
            result = await stream.result()

            # Store messages in database
            self._store_pydanticai_messages(execution.id, result)

            # Extract result data
            result_data = result.data
            if hasattr(result_data, 'model_dump'):
                result_data = result_data.model_dump()

            # Complete execution
            self._complete_execution(execution, {
                "result": result_data,
                "cost": self._extract_cost(result)
            })

            # Yield final result
            yield {
                "type": "complete",
                "result": result_data,
                "cost": self._extract_cost(result),
                "execution_id": execution.id
            }

    def _create_execution(self, agent_id: int, user_id: int, input_data: str) -> Execution:
        """
        Create an Execution record in the database.

        Args:
            agent_id: Agent ID
            user_id: User ID
            input_data: User input

        Returns:
            Created Execution object
        """
        execution = Execution(
            user_id=user_id,
            agent_id=agent_id,
            execution_type='agent',
            input_data={"input": input_data},
            status='running',
            started_at=datetime.now()
        )
        self.session.add(execution)
        self.session.commit()
        self.session.refresh(execution)

        logger.debug(f"Created execution record: {execution.id}")
        return execution

    def _complete_execution(self, execution: Execution, result: Dict[str, Any]):
        """
        Mark execution as completed and store results.

        Args:
            execution: Execution object to update
            result: Execution results
        """
        execution.status = 'completed'
        execution.completed_at = datetime.now()
        execution.output_data = {
            "result": result.get("result", ""),
            "model": result.get("model", ""),
            "cost": result.get("cost")
        }
        self.session.commit()
        logger.debug(f"Execution {execution.id} marked as completed")

    def _fail_execution(self, execution: Execution, error_message: str):
        """
        Mark execution as failed and store error.

        Args:
            execution: Execution object to update
            error_message: Error message
        """
        execution.status = 'failed'
        execution.completed_at = datetime.now()
        execution.error_message = error_message
        self.session.commit()
        logger.debug(f"Execution {execution.id} marked as failed")

    def _store_messages(
        self,
        execution_id: int,
        messages: list,
        message_type: str = "react"
    ):
        """
        Store messages in the database (for ReAct agents).

        Args:
            execution_id: Execution ID
            messages: List of message dicts
            message_type: Type of messages ("react" or "pydanticai")
        """
        for msg in messages:
            message = Message(
                execution_id=execution_id,
                role=msg.get("role", "assistant"),
                content=str(msg.get("content", "")),
                sender=msg.get("sender", msg.get("tool")),
                message_metadata=msg
            )
            self.session.add(message)

        self.session.commit()
        logger.debug(f"Stored {len(messages)} {message_type} messages for execution {execution_id}")

    def _store_pydanticai_messages(self, execution_id: int, result):
        """
        Store PydanticAI messages in the database.

        Args:
            execution_id: Execution ID
            result: PydanticAI result object with all_messages()
        """
        messages = result.all_messages()

        for msg in messages:
            message = Message(
                execution_id=execution_id,
                role=msg.role,
                content=str(msg.content),
                sender="agent" if msg.role == "assistant" else msg.role,
                message_metadata={
                    "timestamp": msg.timestamp.isoformat() if hasattr(msg, 'timestamp') else None,
                    "parts": str(msg.parts) if hasattr(msg, 'parts') else None
                }
            )
            self.session.add(message)

        self.session.commit()
        logger.debug(f"Stored {len(messages)} PydanticAI messages for execution {execution_id}")

    def _format_pydanticai_messages(self, result) -> list:
        """
        Format PydanticAI messages for API response.

        Args:
            result: PydanticAI result object

        Returns:
            List of formatted message dicts
        """
        formatted_messages = []
        for msg in result.all_messages():
            formatted_messages.append({
                "role": msg.role,
                "content": str(msg.content),
                "timestamp": msg.timestamp.isoformat() if hasattr(msg, 'timestamp') else None
            })
        return formatted_messages

    def _extract_cost(self, result) -> Optional[Dict[str, Any]]:
        """
        Extract cost/token usage from PydanticAI result.

        Args:
            result: PydanticAI result object

        Returns:
            Dict with cost information, or None if not available
        """
        try:
            if hasattr(result, 'cost'):
                cost_info = result.cost()
                return {
                    "total_tokens": cost_info.total_tokens if hasattr(cost_info, 'total_tokens') else None,
                    "input_tokens": cost_info.request_tokens if hasattr(cost_info, 'request_tokens') else None,
                    "output_tokens": cost_info.response_tokens if hasattr(cost_info, 'response_tokens') else None,
                    "cost_usd": str(cost_info) if cost_info else None
                }
        except Exception as e:
            logger.warning(f"Could not extract cost information: {e}")
            return None


# Convenience function for direct usage
async def execute_agent_by_id(
    agent_id: int,
    user_id: int,
    input_data: str,
    session: Session,
    stream: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to execute an agent.

    Args:
        agent_id: Agent ID
        user_id: User ID
        input_data: User input
        session: Database session
        stream: Whether to stream responses

    Returns:
        Execution results

    Example:
        >>> from database.database import get_session
        >>> session = get_session()
        >>> result = await execute_agent_by_id(5, 1, "What is 2+2?", session)
    """
    executor = AgentExecutor(session)
    return await executor.execute_agent(agent_id, user_id, input_data, stream)
