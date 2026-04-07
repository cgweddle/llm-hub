# Claude Code Plan: SvelteFlow AI Agent Builder

## Overview
Extend the existing SvelteFlow node builder in `/frontend` to create a visual AI agent builder. Mirror the current setup but replace the sidebar sections (nodes, tools, flows) with **Agent Types** and **Available Tools**.

---

## Phase 1: Define Agent Node Types

Create three custom node components in `src/lib/components/nodes/`:

### 1. PlanningAgentNode.svelte
- **Purpose**: Breaks down tasks into steps
- **Inputs**: Task/goal
- **Outputs**: Step-by-step plan
- **Visual**: Blue themed node

### 2. ReactAgentNode.svelte
- **Purpose**: Reason + Act loop (ReAct pattern)
- **Inputs**: Query, available tools
- **Outputs**: Action results
- **Visual**: Green themed node

### 3. ReflectionAgentNode.svelte
- **Purpose**: Self-critique and improve outputs
- **Inputs**: Content to review
- **Outputs**: Improved content + feedback
- **Visual**: Purple themed node

---

## Phase 2: Update the Canvas

1. Register the new agent node types in the existing SvelteFlow canvas
2. Enable edge connections between agent nodes
3. Add validation for valid agent connection patterns

---

## Phase 3: Update Sidebar

Modify the existing sidebar to have two sections:

### Agent Types
- Planning Agent (draggable)
- React Agent (draggable)
- Reflection Agent (draggable)

### Available Tools
Tools that agents can use (draggable onto agent nodes or as connections):
- Web Search
- Code Executor
- File Reader
- API Caller
- Memory/RAG

Each tool should be assignable to React agents via drag-drop or config panel.

---

## Phase 4: Flow Patterns Support

Enable these agentic patterns through edge connections:

1. **Sequential**: Planning → React → Reflection
2. **Loop**: React ↔ Reflection (iterative improvement)
3. **Parallel**: Multiple React agents from one Planning agent

Add visual indicators for:
- Loop connections (curved edges)
- Data flow direction (animated edges)

---

## Phase 5: Basic Flow Execution (Mock)

1. Extend existing flow execution logic for agent nodes
2. Implement topological sort for execution order
3. Add mock execution that shows data flowing between agents
4. Display execution status on each agent node

---

## File Checklist

Mirror existing structure in `/frontend`:

- [ ] `src/lib/components/nodes/PlanningAgentNode.svelte`
- [ ] `src/lib/components/nodes/ReactAgentNode.svelte`
- [ ] `src/lib/components/nodes/ReflectionAgentNode.svelte`
- [ ] `src/lib/components/sidebar/AgentTypes.svelte`
- [ ] `src/lib/components/sidebar/AvailableTools.svelte`
- [ ] `src/lib/types/agent.ts` - TypeScript types for agents
- [ ] Update existing canvas to register new node types
- [ ] Update existing sidebar to use new sections

---

## Success Criteria

- [ ] Users can drag Planning, React, and Reflection agents onto canvas
- [ ] Users can drag tools from sidebar and assign to agents
- [ ] Agents can be connected via edges
- [ ] Loop patterns are visually supported
- [ ] Mirrors existing `/frontend` structure and patterns