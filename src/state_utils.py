## Get information from the LangGraph State
 
def get_last_message(current_state):
    return current_state.values['messages'][-1]
 
def get_last_message(current_state):
    messages = current_state.values['messages']
    for message in reversed(messages):
        if message.content != "Go to the next step in the process.":
            return message
 
 
def get_last_tool_message(current_state):
    messages = current_state.values['messages']
    for message in reversed(messages):
        if message.type == 'tool':
            return message
    return None
 
def get_last_ai_message(current_state):
    messages = current_state.values['messages']
    for message in reversed(messages):
        if message.type == 'ai':
            return message
    return None
 
def get_next_action(current_state):
    return current_state.next[0]