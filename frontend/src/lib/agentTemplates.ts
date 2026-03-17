/**
 * Agent Templates for the Visual Agent Builder
 * Defines the three core agent types: Planning, React, and Reflection
 */

export interface AgentTemplate {
  type: AgentTypeKey;
  name: string;
  description: string;
  color: string;
  borderColor: string;
  icon: string;
  defaultSystemPrompt: string;
}

export type AgentTypeKey = 'planning' | 'react' | 'reflection' | 'custom';

export const AGENT_TEMPLATES: Record<AgentTypeKey, AgentTemplate> = {
  planning: {
    type: 'planning',
    name: 'Planning Agent',
    description: 'Decomposes complex tasks into smaller, manageable subtasks',
    color: '#3b82f6',      // Blue
    borderColor: '#2563eb',
    icon: '📋',
    defaultSystemPrompt: `You are a Planning Agent responsible for breaking down complex tasks into actionable steps.

Your role is to:
1. Analyze the given task and identify its key components
2. Decompose the task into smaller, logical subtasks
3. Determine the optimal order for executing subtasks
4. Identify dependencies between subtasks
5. Output a clear, structured plan

When creating a plan:
- Be specific and actionable in each step
- Consider potential challenges and edge cases
- Ensure steps are appropriately sized (not too broad, not too granular)
- Include validation checkpoints where appropriate`
  },

  react: {
    type: 'react',
    name: 'React Agent',
    description: 'Reason + Act loop with tool usage for executing tasks',
    color: '#10b981',      // Green
    borderColor: '#059669',
    icon: '⚡',
    defaultSystemPrompt: `You are a React Agent that follows the Reason + Act pattern to accomplish tasks.

Your process:
1. THOUGHT: Analyze the current situation and decide what action to take
2. ACTION: Execute an action using available tools
3. OBSERVATION: Observe the result of the action
4. Repeat until the task is complete

Guidelines:
- Always explain your reasoning before taking action
- Use tools effectively and efficiently
- Handle errors gracefully and adapt your approach
- Provide clear, informative responses
- Know when to ask for clarification vs. making reasonable assumptions`
  },

  reflection: {
    type: 'reflection',
    name: 'Reflection Agent',
    description: 'Self-critiques and improves outputs through iterative refinement',
    color: '#a855f7',      // Purple
    borderColor: '#9333ea',
    icon: '🔍',
    defaultSystemPrompt: `You are a Reflection Agent that critically evaluates and improves outputs.

Your responsibilities:
1. Review the input/previous output for quality and correctness
2. Identify potential issues, gaps, or areas for improvement
3. Suggest specific improvements or corrections
4. When appropriate, provide an improved version

Evaluation criteria:
- Accuracy: Is the information correct?
- Completeness: Are there missing elements?
- Clarity: Is it easy to understand?
- Consistency: Are there contradictions?
- Quality: Does it meet the expected standards?

Always provide constructive feedback with specific suggestions for improvement.`
  },

  custom: {
    type: 'custom',
    name: 'Custom Agent',
    description: 'Fully configurable agent with custom behavior',
    color: '#f59e0b',      // Amber
    borderColor: '#d97706',
    icon: '\u{1f527}',
    defaultSystemPrompt: ''
  }
};

/**
 * Get all agent templates as an array for iteration
 */
export function getAgentTemplatesList(): AgentTemplate[] {
  return Object.values(AGENT_TEMPLATES);
}

/**
 * Get a specific agent template by type
 */
export function getAgentTemplate(type: AgentTypeKey): AgentTemplate {
  return AGENT_TEMPLATES[type];
}

/**
 * Check if a string is a valid agent type key
 */
export function isValidAgentType(type: string): type is AgentTypeKey {
  return type in AGENT_TEMPLATES;
}
