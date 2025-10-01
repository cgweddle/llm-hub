from langgraph.graph import StateGraph
import os
import logging
from logging.handlers import RotatingFileHandler
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



class Graph:
    def __init__(self, nodes, outline):
        """
        outline: dict
            Adjacency list with node names as keys and lists of edges as values.
        """
        self.outline = outline
        self.nodes = nodes
        self.graph = self._build_graph()


    def _build_graph(self):
        # Use a simple state type (dict)
        workflow = StateGraph(dict)
        # Add nodes
        for node in self.outline.keys():
            if node != "START":
                workflow.add_node(self.nodes[node], node)
        
        # Add edges
        for node, edges in self.outline.items():
            for edge in edges:
                workflow.add_edge(node, edge)
        
        return workflow.compile()

