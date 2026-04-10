import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Type

from pydantic import BaseModel, Field, create_model
from pydantic_ai import Agent
from sqlalchemy.orm import Session

from src.database.database import (
    get_evaluation_by_id,
    get_execution_by_id,
    create_evaluation_result,
    update_evaluation_result,
)
import os

logger = logging.getLogger(__name__)

# LangFuse client — imported from agent_executor where it's already initialized
try:
    from src.executors.agent_executor import langfuse_client, LANGFUSE_AVAILABLE
except ImportError:
    LANGFUSE_AVAILABLE = False
    langfuse_client = None


# --- Default structured output models ---

class NumericJudgeResponse(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning for the score")
    score: float = Field(ge=0, le=1, description="Score between 0 and 1")


class BooleanJudgeResponse(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning for the score")
    score: bool = Field(description="True if the output meets the criteria, False otherwise")


class CategoricalJudgeResponse(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning for the score")
    score: str = Field(description="One of the allowed category values")


def _get_response_model(score_type: str, return_fields: Optional[List[str]] = None) -> Type[BaseModel]:
    """Return the appropriate Pydantic model for the given score type and return fields."""
    if not return_fields:
        if score_type == "NUMERIC":
            return NumericJudgeResponse
        elif score_type == "BOOLEAN":
            return BooleanJudgeResponse
        elif score_type == "CATEGORICAL":
            return CategoricalJudgeResponse
        else:
            raise ValueError(f"Unknown score_type: {score_type}")

    # Build dynamic model with custom return fields + score
    fields: Dict[str, Any] = {}
    if score_type == "NUMERIC":
        fields["score"] = (float, Field(ge=0, le=1, description="Score between 0 and 1"))
    elif score_type == "BOOLEAN":
        fields["score"] = (bool, Field(description="True or False"))
    else:
        fields["score"] = (str, Field(description="One of the allowed category values"))

    for field_name in return_fields:
        fields[field_name] = (str, Field(description=f"The {field_name} field"))

    return create_model("JudgeResponse", **fields)


def _get_category_names(score_categories) -> List[str]:
    """Extract category names from structured or simple category lists."""
    if not score_categories:
        return []
    names = []
    for cat in score_categories:
        if isinstance(cat, dict):
            names.append(cat.get("name", str(cat)))
        else:
            names.append(str(cat))
    return names


def _resolve_user_prompt(execution, evaluation) -> str:
    """Substitute {input}, {output}, {tool_output} in the stored user prompt template."""
    template = evaluation.judge_user_prompt or ""
    if not template:
        return "Evaluate the agent's output and provide your score."

    def _format(data):
        if isinstance(data, (dict, list)):
            return json.dumps(data, indent=2)
        return str(data or "")

    result = template
    result = result.replace("{input}", _format(execution.input_data))
    result = result.replace("{output}", _format(execution.output_data))

    # Resolve {tool_output} from child executions
    tool_output_text = ""
    if hasattr(execution, 'children') and execution.children:
        tool_outputs = []
        for child in execution.children:
            if child.execution_type in ('tool_result', 'tool') and child.output_data:
                tool_outputs.append(_format(child.output_data))
        if tool_outputs:
            tool_output_text = "\n---\n".join(tool_outputs)
    result = result.replace("{tool_output}", tool_output_text)

    return result


def _to_numeric(score_type: str, score_value) -> Optional[float]:
    """Convert a judge score to a numeric value for LangFuse."""
    if score_type == "NUMERIC":
        return float(score_value)
    elif score_type == "BOOLEAN":
        return 1.0 if score_value else 0.0
    return None


class EvaluationExecutor:
    def __init__(self, session: Session, llm_config: Optional[Dict[str, Any]] = None):
        self.session = session
        self.llm_config = llm_config or {"models": []}

    async def evaluate(
        self,
        evaluation_id: int,
        execution_id: int,
        user_id: int,
        llm_provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a judge LLM evaluation against an execution.

        Args:
            llm_provider: Optional override. If not provided, uses evaluation.llm_provider.
        """
        evaluation = get_evaluation_by_id(self.session, evaluation_id)
        if not evaluation:
            raise ValueError(f"Evaluation {evaluation_id} not found")

        execution = get_execution_by_id(self.session, execution_id)
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")

        # Create the evaluation result record
        eval_result = create_evaluation_result(
            self.session,
            evaluation_id=evaluation_id,
            execution_id=execution_id,
            user_id=user_id,
            langfuse_trace_id=execution.langfuse_trace_id,
            status="running",
        )

        try:
            provider = llm_provider or evaluation.llm_provider
            model_name = self._resolve_model_name(provider)

            response_model = _get_response_model(
                evaluation.score_type,
                evaluation.return_fields,
            )
            judge_agent = Agent(
                model=model_name,
                system_prompt=evaluation.judge_system_prompt or "",
                output_type=response_model,
            )

            user_message = _resolve_user_prompt(execution, evaluation)
            result = await judge_agent.run(user_message)
            judge_response = result.output

            # Validate categorical score
            if evaluation.score_type == "CATEGORICAL" and evaluation.score_categories:
                allowed = _get_category_names(evaluation.score_categories)
                if judge_response.score not in allowed:
                    raise ValueError(
                        f"Judge returned '{judge_response.score}' which is not in allowed categories: {allowed}"
                    )

            # Post score to LangFuse on the final generation observation
            langfuse_score_id = None
            if LANGFUSE_AVAILABLE and langfuse_client and execution.langfuse_trace_id:
                numeric_value = _to_numeric(evaluation.score_type, judge_response.score)
                reasoning = getattr(judge_response, 'reasoning', None) or str(judge_response)

                # Find the last GENERATION observation (the final LLM output)
                trace = langfuse_client.api.trace.get(execution.langfuse_trace_id)
                generations = [
                    obs for obs in (trace.observations or [])
                    if getattr(obs, 'type', None) == 'GENERATION'
                ]
                if not generations:
                    raise ValueError(f"No GENERATION observations found in trace {execution.langfuse_trace_id}")

                generations.sort(key=lambda o: o.start_time or "", reverse=True)
                observation_id = generations[0].id

                score_kwargs = dict(
                    trace_id=execution.langfuse_trace_id,
                    observation_id=observation_id,
                    name=evaluation.name,
                    comment=reasoning,
                )
                if evaluation.score_type == "NUMERIC":
                    score_kwargs["value"] = numeric_value
                    score_kwargs["data_type"] = "NUMERIC"
                elif evaluation.score_type == "BOOLEAN":
                    score_kwargs["value"] = numeric_value
                    score_kwargs["data_type"] = "BOOLEAN"
                else:
                    score_kwargs["value"] = judge_response.score
                    score_kwargs["data_type"] = "CATEGORICAL"

                score_obj = langfuse_client.create_score(**score_kwargs)
                langfuse_client.flush()
                langfuse_score_id = score_obj.id if hasattr(score_obj, "id") else str(score_obj)

            update_evaluation_result(
                self.session,
                eval_result.id,
                langfuse_score_id=langfuse_score_id,
                status="completed",
                completed_at=datetime.utcnow(),
            )

            return {
                "id": eval_result.id,
                "status": "completed",
                "langfuse_score_id": langfuse_score_id,
            }

        except Exception as e:
            logger.error(f"Evaluation failed for execution {execution_id}: {e}")
            update_evaluation_result(
                self.session,
                eval_result.id,
                status="failed",
                error_message=str(e),
                completed_at=datetime.utcnow(),
            )
            return {
                "id": eval_result.id,
                "status": "failed",
                "error_message": str(e),
            }

    def _resolve_model_name(self, llm_provider: str) -> str:
        """Resolve an LLM provider name using the pre-loaded config dict."""
        for model_config in self.llm_config.get("models", []):
            if model_config.get("name") == llm_provider:
                provider = model_config.get("provider")
                model = model_config.get("model")
                api_key = model_config.get("api_key")
                base_url = model_config.get("base_url")

                if provider == "lmstudio":
                    api_key = api_key or "lm-studio"
                    base_url = base_url or "http://localhost:1234/v1"
                    provider = "openai"

                if api_key:
                    if provider == "anthropic":
                        os.environ["ANTHROPIC_API_KEY"] = api_key
                    else:
                        os.environ["OPENAI_API_KEY"] = api_key
                if base_url:
                    os.environ["OPENAI_BASE_URL"] = base_url

                return f"{provider}:{model}"

        raise ValueError(f"LLM provider '{llm_provider}' not found in config")
