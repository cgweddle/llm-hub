##Import agent types
from langchain.agents import initialize_agent, AgentType, AgentExecutor, create_tool_calling_agent
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool, Tool, tool
from langgraph.types import Command
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage, trim_messages
from pydantic import BaseModel, Field, create_model#, field_validator

import inspect
import operator
import json
from typing import Annotated, Sequence, Literal, List, Any, Optional, Type, Dict
from typing_extensions import TypedDict
import logging
from logging.handlers import RotatingFileHandler

import os
from .llm import CustomLLM



# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create the path for the logs folder in the parent directory
logs_dir = os.path.join(os.path.dirname(current_dir), 'logs')

# Generate a unique log file name with timestamp
log_file_name = f"{os.path.basename(__file__).split('.')[0]}.log"
log_file_path = os.path.join(logs_dir, log_file_name)

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
file_handler = RotatingFileHandler(log_file_path, maxBytes=1485760, backupCount=2)
console_handler = logging.StreamHandler()

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# Disable propagation to avoid duplicate logging
logger.propagate = False


    

class ToolCreator:
    def __init__(self, name, description, func):
        self.name = name
        self.description = description
        self.func = func

    def create_tool(self):
        return Tool(
            name=self.name,
            description=self.description,
            func=self.func
        )


from langchain.chat_models import ChatOpenAI
#from langchain.utilities import PythonREPL
'''
@tool
def generate_function(description: str, input_variables: list, output_variables: list, model:str) -> str:
    """Generate a Python function based on a user description."""
    system_message = "You are a Python coding assistant"
    prompt = f"Write a Python function that {description}. It "
    input_variables_str = "\n".join([f"Input Variable: {name}, Type: {type}" for name, type in input_variables])
    output_variables_str = "\n".join([f"Output Variable: {name}, Type: {type}" for name, type in output_variables])
    prompt = f"Write a Python function that {description}.\n\nIt takes these variables as input:\n{input_variables_str}\n\nIt takes these variables as output:\n{output_variables_str}"
    llm = CustomLLM(system_message=system_message, model=model, stream=False)
    function_code=llm.query_model(prompt=prompt, temperature=.2)
    return function_code


@tool
def decompose_task(task_description: str, model:str) -> list:
    """Decompose a complex task description into smaller subtasks."""
    system_message = "You are an assistant programmer who breaks a complex task into a series of simple substasks so they can be completed in a function."
    prompt = PromptTemplate(
        input_variables=["task_description"],
        template="Decompose the following task description into smaller subtasks:\n\n{task_description}\n\nSubtasks:"
    )
    llm = CustomLLM(system_message=system_message, model=model, stream=True)
    chain = LLMChain(llm=llm, prompt=prompt)
    subtasks = chain.run(task_description)
    return subtasks.split("\n")

@tool
def create_tool(description: str, input_variables: list, output_variables:list,  model: str) -> str:
    """Create a LangChain tool."""
    python_func = generate_function.func(description, input_variables, output_variables, model)
    #Add the @tool decorator
    tool_string = "@tool\n" + python_func
    return tool_string
'''

## Object that will be passed between each node 

class NodeState(TypedDict):
    """
    State of the agent 
    Args:
        messages: past messages from agents
        current_node: name of the current agent
        available_agents: possible next agents based on the graph
        node_types (dict): map the node name to their type
        finished: if the graph should be completed
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_node: str
    available_nodes: List[str]
    finished: bool

'''
class NodeState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
'''


class State(TypedDict):
    messages: Annotated[list, add_messages]



class AgentCreator:
    def __init__(self, name, llm, tools, agent_type=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION):
        self.name = name
        self.llm = llm
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.agent_chain = self.create_agent(self, agent_type, tools)
    def create_agent(self, agent_type, tools):
        return create_react_agent(model=self.llm, tools=tools)
        #return initialize_agent(tools, self.llm, agent=agent_type, verbose=True, memory=self.memory)
    def run(self, task):
        response = self.agent_chain.run(task)
        return response

## Super Class for all possible node types
class Node:
    def __init__(self, name, type):
        self.name = name
        self.type = type

    def invoke(self):
        """This method should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")

    

from pydantic import Extra
def create_internal_tool_input(fields: Dict[str, str]):
    descriptions = {field_name: (str, Field(default=default_value)) for field_name, default_value in fields.items()}
    return create_model('InternalToolInput', **descriptions)

class InternalToolInput(BaseModel):
    """
    Provide to InternalTool object describing the inputs
    Takes arbitrary number of **kwargs, describing the inputs to the tool
    eg. query: The question to search for in the knowledge base

    Args:
        **kwargs: arbitrary number of 
    """
    class Config:
        extra = Extra.allow
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class InternalToolOutput(BaseModel):
    class Config:
        extra = Extra.allow
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class InternalTool(BaseTool):
    name: str 
    description: str 
    args_schema :Optional[Type[BaseModel]] = None
    function: callable 
    preset_args: Dict[str, Any] = Field(default_factory=dict)
    run_arg_descriptions: Dict[str, str] = Field(default_factory=dict)

    
    """
    Preset args: 

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """

    class Config:
        extra=Extra.allow

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args_schema = create_internal_tool_input(fields=self.run_arg_descriptions)
    #def prepare(self, **kwargs):
    #    """
    #    Set the arguments to the values specified in the Agent object **kwargs
    #    """
    #    self.prepared_args = kwargs
    #    return self

    def _run(self, **kwargs: Any) -> Any:
        """Use the tool."""
        combined_args = {**self.preset_args, **kwargs}
        return self.function(**combined_args)

    def _arun(self) -> Any:
        """Use the tool asynchronously."""
        raise NotImplementedError("Async run is not implemented for this tool.")
    
    

class Tool(Node):
    def __init__(self, name, function, **kwargs):
        super().__init__(name, type='tool')
        self.function = function
        self.prepared_args = {}

    def prepare(self, **kwargs):
        """
        Set the arguments to the values specified in the Agent object **kwargs
        """
        self.prepared_args = kwargs
        return self
    
    def invoke(self, state: NodeState) -> NodeState:
        args = self.extract_args(state)
        result = self.function(**args)
        return self.update_state(state, result)
    
    def extract_args(self, state: NodeState) -> dict:
        last_message = state["messages"][-1]
        tool_call = next(call for call in last_message["tool_calls"] if call["function"]["name"] == self.name)
        return json.loads(tool_call["function"]["arguments"])
    
    def update_state(self, state: NodeState, result: str) -> NodeState:
        state["messages"].append({"role": "tool", "content": result})
        return state

def call_agent(
        state: NodeState,
):
    messages = state['messages']
    response = model.invoke()
    

class CustomAgent(Node):
    def __init__(self, name, llm, tools, agent_type, verbose=True, **kwargs):
        """
        Initialize a CustomAgent with unknown extra parameters

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
        super().__init__(name, type='agent')
        self.llm = llm
        self.agent_type = agent_type
        self.verbose = verbose
        self.kwargs = kwargs
        self.tools = load_tools([], llm=llm) + tools
        #self.tools = self._prepare_tools(tools=tools, llm=llm)
        self.agent = create_react_agent(model=self.llm, tools=tools)
        #self.agent = initialize_agent(self.tools, llm, agent=agent_type, verbose=verbose)
    '''
    Depreciated, for defining arguments at the agent level instead of the tool level
    def _prepare_tools(self, tools, llm):
        """
        If a tool has a "prepare" method, 
        set its arguments with the arguments from kwargs that match

        Args:
            tools (_type_): _description_
            llm (_type_): _description_

        Returns:
            _type_: _description_
        """
        prepared_tools = []
        for tool in tools:
            if callable(getattr(tool, 'prepare', None)):
                # Get the parameters of the prepare method
                prepare_params = inspect.signature(tool.prepare).parameters
                # Filter kwargs to only include those that mathc prepare's parameters
                filtered_kwargs = {k: v for k, v in self.kwargs.items() if k in prepare_params}
                prepared_tool = tool.prepare(**filtered_kwargs)
            else:
                prepared_tool = tool
            prepared_tools.append(prepared_tool)
        return load_tools([], llm=llm) + prepared_tools
    '''
    def invoke(self, state):
        """
        Run the agent with the user prompt

        Returns:
            _type_: _description_
        """
        logger.debug(f'Invoking {self.name}')
        #user_message = state["messages"][-1]["content"]      
        #user_message = state["messages"][-1].content
        response = self.agent.invoke(state)
        logger.debug(f'Reuslt of {self.name} invocation: {response}')
        #state["messages"].append(AIMessage(content=response))
        #state["messages"].append({"role": "user", "content": user_message})
        #state["messages"].append({"role": "agent", "content": response})

        #state["current_node"] = self.name

        #if 'END' in response:
        #    finished = True
        #else:
        #    finished = state['finished']
        return response
        '''
        return {"Messages": [HumanMessage(content=response)]}
        
        return Command(
            update={
                "messages": [
                    HumanMessage(content=response["messages"][-1].content, name=self.name)
                ]
            }
        )
        #state["messages"].append({"node": self.name, "content": response})

        return state
        '''
    
    

        


"""
class CustomAgent(Node):
    
    A custom agent class that initializes a language learning model (LLM) agent with specified tools and parameters.

    Attributes:
        tools (list): A list of tools to be loaded for the agent.
        llm: The language learning model to be used by the agent.
        agent_type (str): The type of agent to be initialized.
        verbose (bool): A flag to determine whether the agent should output verbose logs.

    Methods:
        run(query): Runs the agent with the given query and returns the response.

    def __init__(self, name, tools, llm, agent_type, verbose=True):
        super().__init__(name, type='agent')
        self.tools = load_tools([], llm=llm) + [tools]
        self.agent = initialize_agent(self.tools, llm, agent=agent_type, verbose=verbose)
        
    def invoke(self, query):
        response = self.agent.run(query)
        return response
"""
"""
## Class for the nodes of the multiagent graph
class Node:
    def __init__(self, name, agent):
        self.name = name
        self.agent = agent
    def invoke(self, state):
        result = self.agent.invoke(state)
        if isinstance(result, ToolMessage):
            pass
        else:
            result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
        return {
            "messages": [result],
            ## May need to change this if the sender is unknown
            "sender": self.name,
        }
"""


def router(state: NodeState) -> str:
    """
    Function to determine the next node based on a given state
    Args:
        state: Current State of system
        edges: Allowed next nodes of system    
    """
    logger.debug(f'Nodestate: {state.keys()}')
    logger.debug('Starting router')
    messages = state['messages']
    last_message = messages[-1]['messages'][-1]
    logger.debug('Checking available nodes')
    available_nodes = state["available_nodes"]
    logger.debug('Finished checking available nodes')
    if state["finished"]:
        logger.debug('Finished in routing function')
        return "END"
    
    #Check for tool calls
    if "tool_calls" in last_message:
        for tool_call in last_message["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            if tool_name in available_nodes:
                return tool_name

    #Check for agent mentions
    for node in available_nodes:
        logger.debug(f'Available nodes: {node}')
        logger.debug(f'Last Message: {last_message}')
        logger.debug(f'Last message type: {type(last_message)}')
        if node.lower() in last_message.content.lower():
            return node
        
    #If no specific agent is mentioned, return the current agent
    # TODO: make a specific tool that returns the query, with an additional message that none of the available tools or agents were chosen, and to choose only from the available ones
    return state["current_node"]

def agent_function(state: NodeState) -> NodeState:
    agent_name = state["current_node"]
    state["messages"].append({"role": "agent", "content": f"{agent_name}'s response"})
    return state

def tool_function(state: NodeState) -> NodeState:
    tool_name = state["current_node"]
    state["messages"].append({"role": "tool", "content": f"{tool_name} executed"})
    return state

## Incorporate conditoinal edges somehow
#def make_conditional_edge_send(state: OverallState):
#    pass


nodes = {} ## Replace with some function to return nodes at some point
def get_node(name: str):
    return nodes[name]


## Edge types: Normal, Conditoinal, Entry Point, and Conditional Entry Point
## Cnditional edges can include multiple sends with multiple states

def process_input(state, node, outline):
    logger.debug(f'Processing for Node {node.name}')
    logger.debug(f'Input State: {state}')
    #state = node.invoke(state)
    response = node.invoke(state)
    available_nodes = outline[node.name]
    if 'END' in response:
        finished = True
    else:
        finished = False
    logger.debug(f'Output Message: {response}')
    logger.debug(f'Output finished: {str(finished)}')
    return {
        "messages": [response],
        "current_node": node.name,
        "available_nodes": available_nodes,
        "finished": finished
    }
    #return state


class CustomGraph():
    """
    Create a LangChain graph based on a user defined graph described by an adjacency matrix
    
    Attributes
        outline (dict): Adjacency matrix describing the graph's nodes and edges
            Adjacenty matrix keys are node names
                One of them must always be named START, and one of them must always be named END
                Adjacency matrix values are dictionaries of type [node_name]

            Adjacency matrix values are dictionaries of type {"message": message, "edge": {node_name, edge_type}}
    """
    def __init__(self, outline):
        self.outline = outline
        self.graph = None
        logging.debug(f'Outline Keys: {outline.keys()}')

    def _get_node(self, node_name: str, available_nodes: Dict):
        #if node_name == 'START':
        #    return START
        if node_name == 'END':
            return END
        return available_nodes[node_name]

    def create_graph(self, available_nodes):
        workflow = StateGraph(NodeState)
        # Add nodes
        logger.debug('Adding Nodes')

        for node_name in self.outline.keys():
            if node_name == 'START':
                #workflow.set_entry_point(self.outline[node_name][0])
                #logger.debug(f'Entry point set: {self.outline[node_name][0]}')
                #workflow.add_node(START, node_name)
                pass
                '''
                def start_task(state: NodeState):
                    state["messages"].append({"node": "START", "content": "Workflow started."})
                    state["current_node"] = "START"
                    state["available_nodes"] = list(self.outline["START"].keys())
                    state["finished"] = False

                logger.debug("Adding START node")
                workflow.add_node(START, start_task)
                '''


            elif node_name == 'END':
                logger.debug('Adding END node')
                workflow.add_node(END, lambda state: state.update({"finished": True}))
            else:
                logger.debug(f'Getting node {node_name}')
                node = self._get_node(node_name, available_nodes)
                logger.debug(f'Adding node {node_name} to workflow')
                workflow.add_node(node_name, 
                                  lambda state: process_input(state, node=node, outline=self.outline))
        
        start_node = self.outline['START'][0]
        logger.debug(f'Start Node: {start_node}')
        #workflow.set_entry_point(start_node)

        # Add edges
        logger.debug('Adding Edges')
        for node_name, edges in self.outline.items():
            if node_name is not 'START':
                logger.debug(f'Processing edges for {node_name}')
                
                ## Deal with deterministic edges
                if len(edges) == 1:
                    target = edges[0]
                    logger.debug(f'Adding deterministic edge connecting {node_name} to {target}')
                    if node_name == 'START':
                        logger.debug('Adding connecitons to START')
                        workflow.add_edge(START, target)
                    else:
                        logger.debug(f'Dealing with connection to {target}')
                        workflow.add_edge(node_name, END if target == 'END' else target)
                
                ## Deal with conditional edges
                else:
                    logger.debug(f'Adding edge connecting {node_name} to other nodes')
                    edge_map = {edge: END if edge == 'END' else edge for edge in edges}
                    logger.debug('Adding additoinal log info here')
                    logger.debug(f'Adding conditional edges from {node_name}')
                    workflow.add_conditional_edges(
                        node_name,
                        router,
                        edge_map
                    )
            else:
                target = edges[0]
                logger.debug(f'Adding edge from START to {target}')
                workflow.add_edge(START, target)
        
        self.graph = workflow.compile()

        
        ## Add edges

            ## Change to include conditionsl entry points later



class MultiAgentOrchestrator:
    pass




'''
@tool
def split_task(description: str, llm) -> str:
    """Break this command into a set of simpler subtasks. Each subtask should be simple enough to be completed by an LLM."""
    subtasks = llm


@tool
def google_search(query):
    """Search google for relevant websites"""
    pass
@tool
def browse_website(url, prompt):
    """Browse a website to find information"""
    pass
@tool
def analyze_code(code):
    """Give suggestions on how to improve code"""
    pass
'''





        

