#Human in the loop
#Pause the graph at a specific nodes to allow people to check / edit model outputs in the process
 
from langgraph.types import interrupt, Command
from langgraph.graph import MessagesState
from langgraph.prebuilt.interrupt import (
    ActionRequest,
    HumanInterrupt,
    HumanInterruptConfig,
    HumanResponse,
)
from typing import Literal
from .state_utils import *
 
 
 
def human_node(state: MessagesState):
    value = interrupt(
        # Any JSON serializable value to surface to the human.
        # For example, a question or a piece of text or a set of keys in the state
       {
          "text_to_revise": state["some_text"]
       }
    )
    # Update the state with the human's input or route the graph based on the input.
    return {
        "some_text": value
    }
 
 
 
 
## Approve or reject
def create_human_approval_node(approval_node, rejection_node):
    """
    Create an approval node with set approval and rejection nodes
    """
    def human_approval(state: MessagesState) -> Command[Literal[approval_node, rejection_node]]:
        last_message = get_last_message(state)
        is_approved = interrupt(
            {
                'question': "Is this correct?",
                "llm_output": last_message
            }
        )
        if is_approved:
            return Command(goto='approval_node')
        else:
            return Command(goto='rejection_node')
    return human_approval
 
## Review and edit
def human_editing(state: MessagesState):
    result = interrupt(
        {
            "task": "Review the output and make any necessary edits.",
            "llm_output": state["message"] # Correct this
        }
    )
   
    # Update the state with the edited text
    return {
        "llm_output": result["edited_text"]
    }
"""
def create_advanced_human_node(option_routing):
    Create a human node that can route to
 
    Args:
        option_routing (_type_): dictionary with keys "
    def advanced_human_node(state):
"""     
 
def create_human_review_node(option_routing):
    """
    Create a node for human review, that does different things based on specific commands
   
    Args:
        option_routing (dict):
        Keys are user commands, values are node names that the commands route to
            Commands should be "continue", "update", or "feedback"
    """
    options = option_routing.values()
    def human_review_node(state) -> Command[Literal[tuple(options)]]:
        last_message = get_last_ai_message(state)
        tool_call = last_message.tool_calls[-1]
        human_review = interrupt(
            {
                "question": "Is this correct?",
                #Surface tool calls for review
                "tool_call": tool_call
            }
        )
       
        review_action = human_review["action"]
        review_data = human_review.get("data")
 
        # Approval
        if review_action == "continue":
            return Command(goto=option_routing["continue"])
       
        # Modify the tool call manually
        elif review_action == "update":
            ## Update tool call
            updated_message = {
                "role": "ai",
                "content": last_message.content,
                "tool_calls": [
                    {
                        "id": tool_call["id"],
                        "name": tool_call["name"],
                        "args": review_data,
                        "type": "tool_call"
                    }
                ],
                "id": last_message.id
            }
            return Command(goto=option_routing["update"], update={"messages": [updated_message]})
       
        # Give natural langauge feedback, then pass it back to the agent
        elif review_action == "feedback":
            tool_message = {
                "role": "tool",
                "content": review_data,
                "name": tool_call["name"],
                "tool_call_id": tool_call["id"],
            }
            return Command(goto=option_routing["feedback"], update={"messages": [tool_message]})
       
    return human_review_node
 
## Multi-turn conversation
## Sharing human node across multiple agents
 
def create_human_node(options):
    def human_node(state: MessagesState) -> Command[Literal[tuple(options)]]:
        user_input = interrupt(value="Ready for user input.")
       
        # Determine the active agent from the state
        # Can route to the correct agent after collecting input
       
        active_agent = None
       
        return Command(
            update={
                "messages": [{
                    "role": "human",
                    "content": user_input,
                }]
            },
            goto=active_agent
        )
    return human_node
 
## Multi-turn conversation
## Human node per agent
def human_input(state: MessagesState):
    human_message = interrupt("human_input")
    return {
        "messages": [
            {
                "role": "human",
                "content": human_message
            }
        ]
    }