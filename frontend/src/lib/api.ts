// API service for communicating with the backend
const API_BASE_URL = 'http://localhost:8000';

export interface Agent {
  id: number;
  name: string;
  description: string;
  graph_config: AgentGraphConfig;   // Required — unified agent graph
  output_schema?: Record<string, any>;
  is_public: boolean;
  created_at: string;
  updated_at?: string;
}

export interface Tool {
  id: number;
  name: string;
  description: string;
  tool_type: string;
  is_public: boolean;
  created_at: string;
  updated_at: string;
  main_function?: string;
  script_code?: string;
  default_llm?: string;  // Default LLM model name for this tool
}

export interface Flow {
  id: number;
  name: string;
  description: string;
  is_public: boolean;
  created_at: string;
  updated_at: string;
}

export interface User {
  id: number;
  username: string;
  email: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface CondaEnvironment {
  name: string;
  path: string;
}

export interface CondaEnvironmentsResponse {
  status: string;
  message: string;
  environments: CondaEnvironment[];
}

// Flow-related interfaces
export interface NodeConfig {
  node_type: string;  // "tool", "agent", "trigger"
  id: number;
  name: string;
  model_name?: string;  // Reference to LLM config name from ~/.llm_hub/config.yaml
  input_value?: string;  // For trigger nodes — the user's text input
}

export interface EdgeMapping {
  from_node: string;
  to_node: string;
  mapping?: Record<string, string>;  // undefined = passthrough
}

export interface GraphConfig {
  nodes: Record<string, NodeConfig>;
  edges: EdgeMapping[];
  entry_point: string;
  exit_points: string[];
}

export interface FlowCreateRequest {
  name: string;
  description: string;
  graph_config: GraphConfig;
  is_public: boolean;
  user_id: number;
  conda_env?: string;  // Optional conda environment path
}

export interface FlowUpdateRequest {
  name?: string;
  description?: string;
  graph_config?: GraphConfig;
  is_public?: boolean;
  conda_env?: string;
}

export interface FlowExecutionResult {
  flow_id: number;
  execution_id: number | null;
  status: "completed" | "failed";
  final_output: any;
  execution_trace: Array<{
    node: string;
    input: any;
    output: any;
    status: string;
    error?: string;
  }>;
  error?: string;
}

export interface ExecutionDetail {
  id: number;
  parent_id: number | null;
  execution_type: string;
  node_id: string | null;
  name: string | null;
  sequence: number | null;
  input_data: any;
  output_data: any;
  status: string;
  error_message: string | null;
  started_at: string | null;
  completed_at: string | null;
  execution_metadata: Record<string, any> | null;
  langfuse_trace_id: string | null;
  children: ExecutionDetail[];
}

export interface TraceObservation {
  id: string;
  name: string | null;
  type: string | null;
  input: any;
  output: any;
  model: string | null;
  start_time: string | null;
  end_time: string | null;
  usage: { input: number | null; output: number | null; total: number | null } | null;
  level: string | null;
  status_message: string | null;
}

export interface TraceDetail {
  trace_id: string;
  name: string | null;
  input: any;
  output: any;
  observations: TraceObservation[];
}

export interface ExecutionListItem {
  id: number;
  execution_type: string;
  name: string | null;
  status: string;
  started_at: string | null;
  completed_at: string | null;
  flow_id: number | null;
  agent_id: number | null;
}

// API functions - using the new FastAPI endpoints
export async function fetchAvailableAgents(userId: number): Promise<Agent[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/agents/available/${userId}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch available agents: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching available agents:', error);
    return [];
  }
}

export async function fetchAvailableTools(userId: number): Promise<Tool[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/tools/available/${userId}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch available tools: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching available tools:', error);
    return [];
  }
}

export async function fetchAvailableFlows(userId: number): Promise<Flow[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/flows/available/${userId}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch available flows: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching available flows:', error);
    return [];
  }
}

// Individual fetch functions for specific use cases
export async function fetchUserAgents(userId: number): Promise<Agent[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/agents/?user_id=${userId}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch user agents: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching user agents:', error);
    return [];
  }
}

export async function fetchPublicAgents(): Promise<Agent[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/agents/public`);
    if (!response.ok) {
      throw new Error(`Failed to fetch public agents: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching public agents:', error);
    return [];
  }
}

export async function fetchUserTools(userId: number): Promise<Tool[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/tools/user/${userId}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch user tools: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching user tools:', error);
    return [];
  }
}

export async function fetchPublicTools(): Promise<Tool[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/tools/public`);
    if (!response.ok) {
      throw new Error(`Failed to fetch public tools: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching public tools:', error);
    return [];
  }
}

export async function fetchUserFlows(userId: number): Promise<Flow[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/flows/user/${userId}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch user flows: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching user flows:', error);
    return [];
  }
}

export async function fetchPublicFlows(): Promise<Flow[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/flows/public`);
    if (!response.ok) {
      throw new Error(`Failed to fetch public flows: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching public flows:', error);
    return [];
  }
}

// Validation interfaces
export interface ValidationResult {
  compatible: boolean;
  issues: string[];
  compatible_inputs?: string[];
  unsatisfied_required_inputs?: string[];
  output_schema?: any;
  input_schema?: any;
}

export interface ChainValidationResult {
  compatible: boolean;
  issues: string[];
  tool_chain: Array<{
    position: number;
    tool_id: number;
    name: string;
    input_schema: any;
    output_schema: any;
  }>;
}

export interface ConnectionValidationResult {
  compatible: boolean;
  source_field: string;
  target_field: string;
  source_type: string;
  target_type: string;
}

// Validation API functions
export async function validateTwoTools(tool1Id: number, tool2Id: number): Promise<ValidationResult> {
  try {
    const response = await fetch(`${API_BASE_URL}/tools/validate-two`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        tool1_id: tool1Id,
        tool2_id: tool2Id
      })
    });

    if (!response.ok) {
      throw new Error(`Validation failed: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error validating tools:', error);
    return {
      compatible: false,
      issues: [`Validation error: ${error}`]
    };
  }
}

export async function validateConnection(
  tool1Id: number,
  tool2Id: number,
  sourceField: string = "",
  targetField: string = ""
): Promise<ConnectionValidationResult> {
  try {
    const response = await fetch(`${API_BASE_URL}/tools/validate-connection`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        tool1_id: tool1Id,
        tool2_id: tool2Id,
        source_field: sourceField,
        target_field: targetField
      })
    });

    if (!response.ok) {
      throw new Error(`Connection validation failed: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error validating connection:', error);
    return {
      compatible: false,
      source_field: sourceField,
      target_field: targetField,
      source_type: '',
      target_type: ''
    };
  }
}

export async function validateToolChain(toolIds: number[]): Promise<ChainValidationResult> {
  try {
    const response = await fetch(`${API_BASE_URL}/tools/validate-chain`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        tool_ids: toolIds
      })
    });

    if (!response.ok) {
      throw new Error(`Chain validation failed: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error validating tool chain:', error);
    return {
      compatible: false,
      issues: [`Validation error: ${error}`],
      tool_chain: []
    };
  }
}

// User creation
export interface UserCreate {
  username: string;
  email: string;
  password: string;
}

export async function createUser(userData: UserCreate): Promise<User> {
  try {
    const response = await fetch(`${API_BASE_URL}/users/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(userData)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Failed to create user: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error creating user:', error);
    throw error;
  }
}

// Conda environments
export async function fetchCondaEnvironments(): Promise<CondaEnvironment[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/conda/environments`);
    if (!response.ok) {
      throw new Error(`Failed to fetch conda environments: ${response.statusText}`);
    }
    const data: CondaEnvironmentsResponse = await response.json();
    return data.environments;
  } catch (error) {
    console.error('Error fetching conda environments:', error);
    return [];
  }
}

// Flow operations
export async function createFlow(flowData: FlowCreateRequest): Promise<Flow> {
  try {
    // Extract user_id and send it as query param only (not in body)
    const { user_id, ...flowDataWithoutUserId } = flowData;

    const response = await fetch(`${API_BASE_URL}/flows/?user_id=${user_id}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(flowDataWithoutUserId)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Failed to create flow: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error creating flow:', error);
    throw error;
  }
}

export async function updateFlow(flowId: number, flowData: FlowUpdateRequest): Promise<Flow> {
  try {
    const response = await fetch(`${API_BASE_URL}/flows/${flowId}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(flowData)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Failed to update flow: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error updating flow:', error);
    throw error;
  }
}

export async function executeFlow(flowId: number, initialInput: Record<string, any>, condaEnv?: string | null, userId: number = 1): Promise<FlowExecutionResult> {
  try {
    const response = await fetch(`${API_BASE_URL}/flows/${flowId}/execute`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_id: userId,
        initial_input: initialInput,
        conda_env: condaEnv
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Failed to execute flow: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error executing flow:', error);
    throw error;
  }
}

export async function getFlowDetails(flowId: number): Promise<Flow & { graph_config: GraphConfig; conda_env: string | null }> {
  try {
    const response = await fetch(`${API_BASE_URL}/flows/${flowId}`);

    if (!response.ok) {
      throw new Error(`Failed to fetch flow: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching flow details:', error);
    throw error;
  }
}

export async function deleteFlow(flowId: number): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/flows/${flowId}`, {
      method: 'DELETE'
    });

    if (!response.ok) {
      throw new Error(`Failed to delete flow: ${response.statusText}`);
    }
  } catch (error) {
    console.error('Error deleting flow:', error);
    throw error;
  }
}

// Tool creation (generic/manual)
export interface ToolCreate {
  name: string;
  description: string;
  tool_type: string;
  script_code?: string;
  input_schema?: Record<string, any>;
  output_schema?: Record<string, any>;
  is_public: boolean;
}

export async function createTool(userId: number, toolData: ToolCreate): Promise<Tool> {
  try {
    const response = await fetch(`${API_BASE_URL}/tools/?user_id=${userId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(toolData)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Failed to create tool: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error creating tool:', error);
    throw error;
  }
}

// Python script tool creation via factory
export interface PythonScriptToolCreate {
  name: string;
  description: string;
  script_code: string;
  main_function: string;
  is_public: boolean;
  default_llm?: string;  // Optional default LLM model name
}

export async function createPythonScriptTool(userId: number, toolData: PythonScriptToolCreate): Promise<Tool> {
  try {
    const response = await fetch(`${API_BASE_URL}/tools/python-script?user_id=${userId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(toolData)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Failed to create python script tool: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error creating python script tool:', error);
    throw error;
  }
}

// Code generation interfaces
export interface CodeGenerateRequest {
  tool_name: string;
  tool_description: string;
  provider: string;
  model: string;
  api_key?: string;
  base_url?: string;
  additional_instructions?: string;
}

export interface CodeGenerateResponse {
  script_code: string;
  main_function: string;
}

// Code generation API function with streaming
export async function generateToolCodeStream(
  request: CodeGenerateRequest,
  onChunk: (chunk: string) => void,
  onDone: (scriptCode: string, mainFunction: string) => void,
  onError: (error: string) => void
): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/tools/generate-code`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Failed to generate code: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');

      // Keep the last incomplete line in the buffer
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.trim()) {
          try {
            const data = JSON.parse(line);

            if (data.error) {
              onError(data.error);
              return;
            } else if (data.chunk) {
              onChunk(data.chunk);
            } else if (data.done) {
              onDone(data.script_code, data.main_function);
              return;
            }
          } catch (e) {
            console.error('Failed to parse streaming response:', line, e);
          }
        }
      }
    }
  } catch (error) {
    console.error('Error generating code:', error);
    onError(error instanceof Error ? error.message : String(error));
  }
}

// Code editing interfaces
export interface CodeEditRequest {
  existing_code: string;
  editing_instructions: string;
  tool_name: string;
  tool_description: string;
  provider: string;
  model: string;
  api_key?: string;
  base_url?: string;
}

// Code editing API function with streaming
export async function editToolCodeStream(
  request: CodeEditRequest,
  onChunk: (chunk: string) => void,
  onDone: (scriptCode: string, mainFunction: string) => void,
  onError: (error: string) => void
): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/tools/edit-code`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Failed to edit code: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');

      // Keep the last incomplete line in the buffer
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.trim()) {
          try {
            const data = JSON.parse(line);

            if (data.error) {
              onError(data.error);
              return;
            } else if (data.chunk) {
              onChunk(data.chunk);
            } else if (data.done) {
              onDone(data.script_code, data.main_function);
              return;
            }
          } catch (e) {
            console.error('Failed to parse streaming response:', line, e);
          }
        }
      }
    }
  } catch (error) {
    console.error('Error editing code:', error);
    onError(error instanceof Error ? error.message : String(error));
  }
}

// Agent graph types (used by both simple and composed agents)
export interface SubAgentNodeConfig {
  agent_type: string;          // "pydanticai", "react", etc.
  name: string;
  system_prompt: string;
  user_prompt?: string;        // Design-time task instruction, prepended to runtime input
  llm_provider: string;        // Name from config.yaml
  tool_ids: number[];
}

export interface AgentEdgeConfig {
  from_node: string;
  to_node: string;
  is_loop: boolean;
}

export interface AgentGraphConfig {
  nodes: Record<string, SubAgentNodeConfig>;
  edges: AgentEdgeConfig[];
  entry_point: string;
  exit_points: string[];
  max_loop_iterations?: number;
}

// Agent creation
export interface AgentCreateData {
  name: string;
  description: string;
  graph_config: AgentGraphConfig;       // Required — unified agent graph
  output_schema?: Record<string, any>;
  is_public?: boolean;
}

export async function createAgent(userId: number, agentData: AgentCreateData): Promise<Agent> {
  try {
    const response = await fetch(`${API_BASE_URL}/agents/?user_id=${userId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(agentData)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Failed to create agent: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error creating agent:', error);
    throw error;
  }
}

export interface AgentUpdateData {
  name?: string;
  description?: string;
  graph_config?: AgentGraphConfig;
  output_schema?: Record<string, any>;
}

export async function updateAgent(agentId: number, updateData: AgentUpdateData): Promise<Agent> {
  try {
    const response = await fetch(`${API_BASE_URL}/agents/${agentId}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(updateData)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Failed to update agent: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error updating agent:', error);
    throw error;
  }
}

// System prompt generation interfaces
export interface SystemPromptGenerateRequest {
  agent_name: string;
  agent_description: string;
  tool_names: string[];
  model: string;
  additional_instructions?: string;
}

// System prompt generation with streaming
export async function generateSystemPromptStream(
  request: SystemPromptGenerateRequest,
  onChunk: (chunk: string) => void,
  onDone: (systemPrompt: string) => void,
  onError: (error: string) => void
): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/agents/generate-system-prompt`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Failed to generate system prompt: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');

      // Keep the last incomplete line in the buffer
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.trim()) {
          try {
            const data = JSON.parse(line);

            if (data.error) {
              onError(data.error);
              return;
            } else if (data.chunk) {
              onChunk(data.chunk);
            } else if (data.done) {
              onDone(data.system_prompt);
              return;
            }
          } catch (e) {
            console.error('Failed to parse streaming response:', line, e);
          }
        }
      }
    }
  } catch (error) {
    console.error('Error generating system prompt:', error);
    onError(error instanceof Error ? error.message : String(error));
  }
}

// User prompt generation with streaming
export async function generateUserPromptStream(
  request: SystemPromptGenerateRequest,
  onChunk: (chunk: string) => void,
  onDone: (userPrompt: string) => void,
  onError: (error: string) => void
): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/agents/generate-user-prompt`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Failed to generate user prompt: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');

      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.trim()) {
          try {
            const data = JSON.parse(line);

            if (data.error) {
              onError(data.error);
              return;
            } else if (data.chunk) {
              onChunk(data.chunk);
            } else if (data.done) {
              onDone(data.user_prompt);
              return;
            }
          } catch (e) {
            console.error('Failed to parse streaming response:', line, e);
          }
        }
      }
    }
  } catch (error) {
    console.error('Error generating user prompt:', error);
    onError(error instanceof Error ? error.message : String(error));
  }
}

// LLM Provider Config interfaces
export interface LLMProviderConfig {
  name: string;
  provider: string;  // 'anthropic' | 'openai' | 'gemini' | 'lmstudio'
  api_key?: string;
  base_url?: string;
  model: string;
}

export interface LLMProvidersConfigResponse {
  models: LLMProviderConfig[];
}

export interface LLMProvidersConfigRequest {
  models: LLMProviderConfig[];
}

// LLM Provider Config API functions
export async function loadLLMProvidersConfig(): Promise<LLMProvidersConfigResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/llm-providers/config`);

    if (!response.ok) {
      throw new Error(`Failed to load LLM config: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error loading LLM providers config:', error);
    // Return empty config on error
    return {
      models: []
    };
  }
}

export async function saveLLMProvidersConfig(config: LLMProvidersConfigRequest): Promise<{ status: string; message: string; config_path: string }> {
  try {
    const response = await fetch(`${API_BASE_URL}/llm-providers/config`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Failed to save LLM config: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error saving LLM providers config:', error);
    throw error;
  }
}

// ─── Execution API functions ───

export async function fetchExecutions(userId: number, limit: number = 50, offset: number = 0): Promise<ExecutionListItem[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/executions?user_id=${userId}&limit=${limit}&offset=${offset}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch executions: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching executions:', error);
    return [];
  }
}

export async function fetchExecution(executionId: number): Promise<ExecutionDetail | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/executions/${executionId}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch execution: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching execution:', error);
    return null;
  }
}

export async function fetchExecutionTrace(executionId: number): Promise<TraceDetail | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/executions/${executionId}/trace`);
    if (!response.ok) {
      if (response.status === 404) return null;
      throw new Error(`Failed to fetch trace: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching execution trace:', error);
    return null;
  }
}
