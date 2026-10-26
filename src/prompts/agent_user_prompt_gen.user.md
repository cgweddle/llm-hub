Write a user prompt template for an AI agent with the following details:

Agent Name: {AGENT_NAME}
Agent Role: {AGENT_DESCRIPTION}

{TOOLS_SECTION}

{ADDITIONAL_SECTION}

The agent's system prompt is:
---
{SYSTEM_PROMPT}
---

Write a concise user prompt template that complements this system prompt. Use at least one named input such as {input} to reference the runtime input, with a separate descriptive name for each distinct piece of incoming data. Only use {message_history} if the agent's role requires awareness of previous conversation context.

Write the user prompt template now.