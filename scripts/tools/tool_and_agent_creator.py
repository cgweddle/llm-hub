from langchain.tools import BaseTool, Tool, tool, StructuredTool
from langgraph.graph import MessagesState, START, END
from typing import Annotated, Sequence, Callable, Literal, get_type_hints, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
#from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_tool_calling_agent
from pydantic import BaseModel
import inspect
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
import os
os.environ["LANGGRAPH_DEBUG"] = "0"
import logging
 
logging.basicConfig(filename='../logs/tool_creator.log',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
 
logger = logging.getLogger(__name__)
 
class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
 
class AgentNode:
    def __init__(self, llm, system_prompt):
        self.llm = llm
        self.system_prompt = system_prompt
 
    def invoke(self, user_prompt: str, state: MessagesState):
        """
        Invoke the agent to query the LLM with the user prompt and maintain state.
 
        Args:
            user_prompt (str): Prompt from the user.
            state (MessagesState): The current state of messages.
 
        Returns:
            The LLM's response.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ] + state["messages"]
        response = self.llm.invoke(messages)
        return response
 
 
 
 
def create_agent_only_use_tools(llm, tools):
    """
    Only call the tools available, do not include an LLM chatbot
    Allows for multiple tools to be used, and chained tool use,
    where the output for one tool is used as the input to another
    """
    llm_with_tools = llm.bind_tools(tools)
    def agent_tool_executor(state: AgentState):
        messages = state["messages"]
        while True:
            logger.debug('Invoking llm with tools')
            response = llm_with_tools.invoke(messages)
            logger.debug(f'Response here: {response}')
            if not response.tool_calls:
                logger.debug('Returning response')
                return {"messages": [response]}
           
            # Process the tool calls
            logger.debug('Processing tool calls')
            tool_results = []
            for tool_call in response["tool_calls"]:
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                if tool_name in tools:
                    tool_result = tools[tool_name](**tool_args)
                    tool_results.append({"tool": tool_name, "result": tool_result})
                else:
                    logger.warning(f"Unknown tool requested: {tool_name}")
           
            messages.append({"role": "assistant", "content": str(tool_results)})
   
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", agent_tool_executor)
    graph_builder.set_entry_point("agent")
    logger.debug("Compiling graph")
    return graph_builder.compile()
 
class Node:
    def __init__(self, name, system_message, llm):
        self.name = name
        self.system_message = system_message
        self.llm = llm
    def invoke(self, state):
        messages = [
            {"role": "system", "content": self.system_message}
            ] + [state["messages"]]
        return self.llm.invoke(messages)
       
 
def create_llm_only_node(llm, system_message):
    """
    If there are tool calls in the messages, Anthropic requests a tool be defined
    """
    def call_llm(state: List):
        messages = [
            {"role": "system", "content": system_message}
            ] + state["messages"]
        return llm.invoke(messages)
    llm_tool = create_tool(name='llm_call', description='Call the LLM', func=call_llm)
    tool_agent = create_tool_calling_agent(llm, tools=[llm_tool], prompt='')
    return tool_agent
 
 
 
 
class CustomAgent():
    def __init__(self, name, description, llm, tools, **kwargs):
        """
        Initialize a CustomAgent with unknown extra parameters
 
        ***Right now it just works with one tool at a time***
        Args:
            name (str): The name of the agent
            llm (Any): Langchain-compatible LLM to use
            tools (List[Any]): A list of tools the agent can use
            agent_type (AgentType): LangChain Agent Type
            **kwards: Additional arguments to be stored and used by the agent
 
        Raises:
            NotImplementedError: _description_
 
        Returns:
            _type_: _description_
        """
        self.llm = llm
        self.name = name
        self.tools = tools
        self.llm_with_tools = llm.bind_tools(tools)
        self.description = description
        self.kwargs = kwargs
        #self.tools = load_tools([], llm=llm) + tools
   
    def create_prompt(self):
        tools_str = "\n".join([f"{tool.name} tool: {tool.description}" for tool in self.tools])
        agent_prompt = f"You are an AI agent tasked with {self.description}. "
        agent_prompt += f"You have access to the following tools: \n{tools_str}\n"
        agent_prompt += ("Do not ask for clarification. "
                        "Respond only with the result of the tool. Do not add additional text or information. "
                        "Only address the parts of the prompt that fall within your capabilities. "
                        "If a task is outside your description or the scope of your tools, do not attempt to answer it. "
                        "Simply end and allow a different agent to handle that part of the request. "
                        "Do not add extra text or information about why you are ending.")
        return agent_prompt
    def invoke(self, state):
        logger.debug(f'Invoking {self.name}')
        agent_prompt = self.create_prompt()
        print(f'Custom Agent Prompt: {agent_prompt}')
        print(f'Available tools: {', '.join([t.name for t in self.tools])}')
        messages = state["messages"] +  [{'role': 'user', 'content': agent_prompt}]
        response = self.llm_with_tools.invoke(messages)
        messages.append(response)
        print(f"LLM response: {response}")
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print('Tool calls: ', response.tool_calls)
           
            # Iterate over each tool call in the response
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args")
                tool_id = tool_call.get("id")
               
                # Find the corresponding tool by name
                tool = next((t for t in self.tools if t.name == tool_name), None)
                if tool:
                    # Execute the tool with the provided arguments
                    tool_output = tool.invoke(tool_args)
                   
                    # Assuming ToolMessage is a class or function, make sure it's correctly defined or imported
                    tool_message = ToolMessage(
                        content=tool_output,
                        tool_call_id=tool_call["id"]
                    )
                   
                    # Append the tool's output as a new message
                    messages.append(tool_message)
                    return {"messages": messages}
   
                else:
                    return {"messages": [response, {'role': 'user', 'content': 'Tool did not run correctly'}]}
        else:
            return {"messages": messages}
        """
        tool_outputs = {}
        tool_call_ids = []
        while True: 
            print('messages: ', messages)
            response = self.llm_with_tools.invoke(messages) 
            print('response:', response)  # Debugging output
 
            if response.tool_calls:
                print('tool calls: ', response.tool_calls)
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]
 
                    # Inject previous tool results into the tool arguments
                    if isinstance(tool_args, dict):
                        tool_args = {k: (tool_outputs[v] if v in tool_outputs else v) for k, v in tool_args.items()}
 
                    # Find and execute the tool
                    tool = next((t for t in self.tools if t.name == tool_name), None)
                    if tool:
                        tool_output = tool.invoke(tool_args)
                        print('tool output: ', tool_output)
                        tool_outputs[tool_name] = tool_output  # Store tool output for dependencies
                        tool_call_ids.append(tool_call.get("id"))
                        # Append the tool result back into the message history
 
                        tool_message = ToolMessage(
                            content = tool_output,
                            tool_call_id = tool_call["id"]
                        )
                        messages.append(tool_message)
           
            else:
                # If there are no tool calls, return the final response
                return response.content
        """
       
 
def create_agent(llm, type, tools, agent_description):
    def generate_prompt(tools, agent_description):
        tools_str = "\n".join([f"{tool.name} tool: {tool.description}" for tool in tools])
        agent_prompt = f"You are an AI agent tasked with {agent_description}. "
        agent_prompt += f"You have access to the following tools: \n{tools_str}\n"
        agent_prompt += ("Do not ask for clarification. "
                        "Respond only with the result of the tool. Do not add additional text or information. "
                        "Only address the parts of the prompt that fall within your capabilities. "
                        "If a task is outside your description or the scope of your tools, do not attempt to answer it. "
                        "Simply end and allow a different agent to handle that part of the request. "
                        "Do not add extra text or information about why you are ending.")
        return agent_prompt
   
    agent_prompt = generate_prompt(tools=tools, agent_description=agent_description)
   
    if type == 'react':
        return create_react_agent(llm, tools=tools, prompt=agent_prompt)
    elif type == 'tool_calling':
        return create_tool_calling_agent(llm, tools=tools, prompt=agent_prompt)
    else:
        return None
    """
    def agent_wrapper(state):
        messages = state["messages"]
 
        # If the assistant was the last speaker and no tool was used, prevent an invalid state
        if messages[-1]["role"] == "assistant":
            messages.append({"role": "user", "content": "Okay, what's next?"})
 
        return agent(state)
    """
 
 
    """
    logger.debug('Starting agent creator')
    llm_with_tools = llm.bind_tools(tools)
   
    def chatbot(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", chatbot)
 
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    logger.debug('Adding edge')
    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
    )
    graph_builder.add_edge("tools", "agent")
    graph_builder.set_entry_point("agent")
    logger.debug('Compiling graph')
    graph = graph_builder.compile()
    logger.debug('Returning graph')
    return graph
    """
 
def create_tool(name: str, description: str, func: Callable):
    type_hints = get_type_hints(func)
    input_model = type(
        f"{name}Inputs",  # Dynamically name the model
        (BaseModel,),
        {"__annotations__": type_hints}  # Ensure correct type annotations
    )
    return StructuredTool(name=name, description=description, func=func, args_schema=input_model)  
    """
    class_body = {
        '__annotations__': {
            'name': str,  # Type annotation for 'name'
            'description': str,  # Type annotation for 'description'
            'run': Callable,  # Type annotation for 'run'
            'arun': Callable,  # Type annotation for 'arun'
        },
        'name': name,
        'description': description,
        'run': func,
        'arun': func
    }
    tool_class = type(name, (Tool,), class_body)
    return tool_class
    """
   
def create_router(options: list[str]):
    NextOption = Literal[tuple(options)]
    class Router(TypedDict):
        next: NextOption
    return Router
   
def create_supervisor_node(llm, nodes):
    end = False
    if "__end__" in nodes:
        nodes.remove("__end__")
        end = True
    print(nodes)
    options = [n.keywords["name"] for n in nodes]
    node_description_str = '\n'.join([f"{node.keywords["name"]} Node: {node.keywords["description"]}\n\tTools:{"\n\t\t".join([f"{t["name"]}: {t["description"]}" for t in node.keywords["tool_descriptions"]])}" for node in nodes if node != '__end__'])
    if end:
        options.append("__end__")
    def supervisor_node(state: MessagesState) -> Command[Literal[tuple(options)]]:       
        """
        system_prompt = (
            "You are a supervisor tasked with managing a conversation between the"
            f" following workers: {options}. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status."
            f"{" When finished, respond with FINISH." if "__end__" in options else ""}"
        )
        """
        """
        system_prompt = (
            "You are a supervisor managing a team of specialized agents: "
            f"{options}. Each agent is trained for a specific task and should "
            "only perform that task. Your job is to route the conversation to the "
            "appropriate agent at the correct time."
            "\n\nRules:"
            "\n- Assign each request only to the agent specialized in handling it."
            "\n- Agents should not attempt tasks outside their expertise."
            "\n- If multiple tasks are required, route the conversation step by step."
            f"{"\n- When all tasks are completed, return 'FINISH'." if "__end__" in options else ""}"
        )
        """
        system_prompt = (
            "You are a supervisor managing a team of specialized agents: "
            f"{options}. Each agent is trained for a specific task and should "
            "only perform that task. Your job is to route the conversation to the "
            "appropriate agent at the correct time."
            f"\n The following is a description of each of the agents and the tools avilable to that agent: {node_description_str}"
            "\n\nRules:"
            "\n- Assign each request only to the agent specialized in handling it."
            "\n- Agents should not attempt tasks outside their expertise."
            "\n- If multiple tasks are required, route the conversation step by step."
            f"{"\n- When all tasks are completed, return 'FINISH'." if end else ""}"
        )
        print('system prompt: ', system_prompt)
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
       
        Router = create_router(options=options)
        print('messages: ', messages)
        response = llm.with_structured_output(Router).invoke(messages)
        print('response: ', response)
        goto = response["next"]
        print(f"Next Worker: {goto}")
        if goto == "FINISH" and "__end__" in options:
            goto=END
        return Command(goto=goto)
    return supervisor_node