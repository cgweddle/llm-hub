"""
Tool Compatibility Validation
Validates if tools can work together in a workflow by checking type compatibility
"""

import sys
import os
import logging
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database.database import get_session
from database.database_setup import Tool

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def validate_tool_compatibility(tool_ids: List[int]) -> Dict[str, Any]:
    """
    Validate that tools can work together in a workflow

    Args:
        tool_ids: List of tool IDs in the order they would be chained

    Returns:
        Dict with compatibility information including:
        - compatible: bool indicating if all tools are compatible
        - issues: list of compatibility issues found
        - tool_chain: list of tool information
    """
    compatibility_report = {
        "compatible": True,
        "issues": [],
        "tool_chain": []
    }

    session = get_session()
    try:
        prev_tool_id = None

        for i, tool_id in enumerate(tool_ids):
            tool = session.query(Tool).filter(Tool.id == tool_id).first()

            if not tool:
                compatibility_report["compatible"] = False
                compatibility_report["issues"].append(f"Tool with ID {tool_id} not found")
                continue

            compatibility_report["tool_chain"].append({
                "position": i,
                "tool_id": tool_id,
                "name": tool.name,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema
            })

            # Check compatibility with previous tool using validate_two_tools
            if prev_tool_id is not None:
                result = validate_two_tools(prev_tool_id, tool_id)
                if not result["compatible"]:
                    compatibility_report["compatible"] = False
                    compatibility_report["issues"].extend(result["issues"])

            # Save current tool ID as previous for next iteration
            prev_tool_id = tool_id

    finally:
        session.close()

    return compatibility_report


def validate_two_tools(tool1_id: int, tool2_id: int) -> Dict[str, Any]:
    """
    Validate if tool1's output is compatible with tool2's input
    Checks strict compatibility - all required inputs of tool2 must be satisfiable

    Args:
        tool1_id: ID of first tool (output provider)
        tool2_id: ID of second tool (input consumer)

    Returns:
        Dict with compatibility information
    """
    session = get_session()
    try:
        tool1 = session.query(Tool).filter(Tool.id == tool1_id).first()
        tool2 = session.query(Tool).filter(Tool.id == tool2_id).first()

        if not tool1 or not tool2:
            return {"compatible": False, "error": "Tool not found"}

        output_schema = tool1.output_schema or {}
        input_schema = tool2.input_schema or {}

        # Strict compatibility check
        compatibility_issues = []
        output_type = output_schema.get("type")

        compatible_inputs = []
        unsatisfied_required_inputs = []

        for param_name, param_info in input_schema.items():
            param_type = param_info.get("type")
            is_optional = param_info.get("optional", False)

            if param_type == output_type:
                compatible_inputs.append(param_name)
            elif not is_optional:
                # Required parameter that can't be satisfied by tool1's output
                unsatisfied_required_inputs.append(f"{param_name}: {param_type}")

        if not compatible_inputs:
            # Output doesn't match ANY input parameter
            compatibility_issues.append(
                f"Tool {tool1_id} ({tool1.name}) output type '{output_type}' cannot feed into any input of tool {tool2_id} ({tool2.name})"
            )

        if unsatisfied_required_inputs:
            # Required inputs remain unsatisfied
            compatibility_issues.append(
                f"Tool {tool2_id} ({tool2.name}) has required inputs that cannot be satisfied by tool {tool1_id} ({tool1.name}): {', '.join(unsatisfied_required_inputs)}"
            )

        return {
            "compatible": len(compatibility_issues) == 0,
            "issues": compatibility_issues,
            "compatible_inputs": compatible_inputs,
            "unsatisfied_required_inputs": unsatisfied_required_inputs,
            "output_schema": output_schema,
            "input_schema": input_schema
        }

    finally:
        session.close()



def validate_connection(
        tool1_id: int,
        tool2_id: int,
        source_field: str,
        target_field: str
) -> Dict[str, Any]:
    """
    Validate specific connection between two tools

    Args:
        tool1_id: ID of source tool
        tool2_id: ID of target tool
        source_field: field name from source node
        target_field: field name from target node

    Returns:
        Dict with:
            - "compatible": bool
            - source_field: output field being connected (None for whole output)
            - target_field: input parameter being connected
            - source_type: type of the source field
            - target_type: type of the target field
    """
    logger.debug(f"validate_connection called: tool1_id={tool1_id}, tool2_id={tool2_id}, source_field='{source_field}', target_field='{target_field}'")
    session = get_session()

    try:
        tool1 = session.query(Tool).filter(Tool.id == tool1_id).first()
        tool2 = session.query(Tool).filter(Tool.id == tool2_id).first()

        logger.debug(f"tool1: {tool1.name if tool1 else None}, tool2: {tool2.name if tool2 else None}")

        if not tool1 or not tool2:
            logger.error("Tool not found")
            return {"compatible": False, "issues": ["Tool not found"]}
    
        output_schema = tool1.output_schema
        input_schema = tool2.input_schema

        logger.debug(f"output_schema: {output_schema}")
        logger.debug(f"input_schema: {input_schema}")

        ## If no source_field, get the type of the whole output
        if source_field == "":
            logger.debug("Using whole output (source_field is empty)")
            source_type = output_schema.get("type") if output_schema else None
            logger.debug(f"source_type from whole output: {source_type}")
        else:
            logger.debug(f"Using specific field: {source_field}")
            if output_schema and output_schema.get("type") == "dict":
                source_properties = output_schema.get("properties", {})
                logger.debug(f"source_properties: {list(source_properties.keys())}")
                if source_field in source_properties:
                    source_type = source_properties[source_field].get("type")
                    logger.debug(f"source_type: {source_type}")
                else:
                    logger.error(f"Field '{source_field}' not in source_properties")
                    raise ValueError(f"No {source_field} in output schema properties")
            else:
                logger.error(f"Output schema type is {output_schema.get('type') if output_schema else None}, not dict")
                raise ValueError(
                    f"Cannot access field '{source_field}' on non-dict output"
                )

        ## Get target params
        logger.debug(f"Getting target type for field: '{target_field}'")
        # Input schema format: {"param_name": {"type": "str", "optional": false}}
        # Parameters are at the top level, not under "properties"
        if input_schema:
            logger.debug(f"input_schema parameters: {list(input_schema.keys())}")
        else:
            logger.warning("input_schema is None")
            input_schema = {}

        if target_field in input_schema:
            target_type = input_schema[target_field].get("type")
            logger.debug(f"target_type: {target_type}")
        else:
            logger.error(f"Field '{target_field}' not in input_schema. Available: {list(input_schema.keys())}")
            raise ValueError(f"Parameter '{target_field}' not found in input schema. Available: {list(input_schema.keys())}")

        logger.debug(f"Comparing types: {source_type} == {target_type}")
        if source_type == target_type:
            compatible = True
        else:
            compatible = False

        logger.debug(f"Validation result: compatible={compatible}")
        return {
            "compatible": compatible,
            "source_field": source_field,
            "target_field": target_field,
            "source_type": source_type,
            "target_type": target_type
        }

    except Exception as e:
        logger.exception(f"Error in validate_connection: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Validate tool compatibility")
    parser.add_argument("tool_ids", nargs="+", type=int, help="Tool IDs to validate")
    args = parser.parse_args()

    result = validate_tool_compatibility(args.tool_ids)

    print(f"\nCompatibility Report:")
    print(f"Compatible: {result['compatible']}")

    if result['issues']:
        print(f"\nIssues found:")
        for issue in result['issues']:
            print(f"  - {issue}")

    print(f"\nTool Chain:")
    for tool in result['tool_chain']:
        print(f"  {tool['position']}. {tool['name']} (ID: {tool['tool_id']})")
        print(f"     Input: {tool['input_schema']}")
        print(f"     Output: {tool['output_schema']}")
