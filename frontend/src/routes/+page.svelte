<script lang="ts">
  import { writable } from 'svelte/store';
  import {
    SvelteFlow,
    Background,
    Controls,
    MiniMap,
    Position,
    type Node,
    type Edge,
    addEdge,
    ConnectionLineType,
    MarkerType,
    type Viewport
  } from '@xyflow/svelte';
  import { EditorView, basicSetup } from 'codemirror';
  import { python } from '@codemirror/lang-python';
  import { oneDark } from '@codemirror/theme-one-dark';
  import { EditorState } from '@codemirror/state';
  import { onDestroy, setContext, tick, untrack } from 'svelte';

  import ColorSelectorNode from './ColorSelectorNode.svelte';
  import ToolNode from './ToolNode.svelte';
  import TriggerNode from './TriggerNode.svelte';
  import FloatingEdge from './FloatingEdge.svelte';
  import AgentBuilder from './AgentBuilder.svelte';
  import type { AgentTemplate } from '$lib/agentTemplates';
  import CondaEnvironmentsPanel from './CondaEnvironmentsPanel.svelte';
  import LLMProvidersPanel from './LLMProvidersPanel.svelte';
  import EvaluationManager from './EvaluationManager.svelte';
  import FullscreenNodeModal from './FullscreenNodeModal.svelte';
  import InfoPanel from './InfoPanel.svelte';
  import { fullscreenNode } from '$lib/stores/fullscreenNode';
  import { builderMode, type BuilderMode } from '$lib/stores/builderMode';
  import { Button } from "$lib/components/ui/button";
  import { Input } from "$lib/components/ui/input";
  import { Label } from "$lib/components/ui/label";
  import {
    validateTwoTools,
    validateConnection,
    createFlow,
    updateFlow,
    executeFlow,
    getFlowDetails,
    deleteFlow,
    deleteAgent,
    deleteTool,
    createPythonScriptTool,
    generateToolCodeStream,
    editToolCodeStream,
    createAgent,
    generateSystemPromptStream,
    generateUserPromptStream,
    type ValidationResult,
    type ConnectionValidationResult,
    type Tool,
    type Agent,
    type FlowCreateRequest,
    type FlowUpdateRequest,
    type Flow as FlowType,
    type CodeGenerateRequest,
    type AgentCreateData,
    type SystemPromptGenerateRequest,
    fetchExecution,
    evaluateExecution
  } from '../lib/api';
  import { buildEnhancedGraphConfig } from '$lib/flowBuilder';
  import { autoLayoutNodes } from '$lib/elkLayout';
  import '@xyflow/svelte/dist/style.css';
  import type { PageData } from './$types';

  let { data } = $props<{ data: PageData }>();

  // Track viewport for coordinate conversion
  let viewport: Viewport = $state({ x: 0, y: 0, zoom: 1 });

  // Validation state
  let validationMessage = $state('');
  let showValidationToast = $state(false);
  let validationSuccess = $state(false);
  let previousEdgeCount = $state(0);

  // Watch for edge deletions and dismiss validation toast
  $effect(() => {
    if (edges.length < previousEdgeCount && showValidationToast && !validationSuccess) {
      console.log('Edge was deleted, dismissing validation toast');
      showValidationToast = false;
    }
    previousEdgeCount = edges.length;
  });

  // Flow save state
  let flowName = $state('');
  let flowDescription = $state('');
  let showSaveDialog = $state(false);
  let isSaving = $state(false);

  // Current flow tracking
  let currentFlowId: number | null = $state(null);

  // Info panel state
  let showInfoPanel = $state(false);
  let lastExecutionId: number | null = $state(null);
  let evalsEnabled = $state(false);
  let evalsRunning = $state(false);

  // Conda environment state
  let selectedCondaEnv: string | null = $state(null);

  // LLM provider state
  import type { LLMProvider } from '$lib/store';
  let selectedLLMProvider: LLMProvider | null = $state(null);
  let llmProviders: LLMProvider[] = $state([]);

  // Make llmProviders available to child components (ToolNode) as a writable store
  const llmProvidersStore = writable<LLMProvider[]>([]);
  setContext('llmProviders', llmProvidersStore);

  // Sync llmProviders array with the store
  $effect(() => {
    llmProvidersStore.set(llmProviders);
  });

  // Create Tool modal state
  let showCreateToolModal = $state(false);
  let newToolName = $state('');
  let newToolDescription = $state('');
  let newToolCode = $state('');
  let newToolMainFunction = $state('');
  let newToolIsPublic = $state(false);
  let showWriteWithAI = $state(false);
  let additionalInstructions = $state('');
  let showEditWithAI = $state(false);
  let editingInstructions = $state('');
  let isCreatingTool = $state(false);
  let isGeneratingCode = $state(false);
  let isEditingCode = $state(false);

  // Create Agent modal state
  let showCreateAgentModal = $state(false);
  let newAgentName = $state('');
  let newAgentDescription = $state('');
  let newAgentSystemPrompt = $state('');
  let newAgentUserPrompt = $state('{input}');
  let newAgentSelectedTools: number[] = $state([]);
  let newAgentSelectedEvals: number[] = $state([]);
  let newAgentLLMProvider = $state('');
  let isCreatingAgent = $state(false);
  let showGeneratePromptAI = $state(false);
  let promptAdditionalInstructions = $state('');
  let isGeneratingPrompt = $state(false);
  let newAgentOutputPaths: Array<{name: string, description: string, return_behavior: string}> = $state([]);
  let createAgentUserPromptBackdrop: HTMLDivElement = $state(undefined as any);
  let isConfiguringComplexNode = $state(false);
  let pendingAgentTemplate: AgentTemplate | null = $state(null);
  let agentBuilderRef: AgentBuilder = $state(undefined as any);

  // Agent Builder mode
  let currentMode: BuilderMode = $state('flow');
  builderMode.subscribe(mode => currentMode = mode);

  // Track fullscreen node state
  let currentFullscreenNode = $state<import('$lib/stores/fullscreenNode').FullscreenNodeData | null>(null);
  fullscreenNode.subscribe(value => currentFullscreenNode = value);

  // Intercept fullscreen opens for complex agents — redirect to AgentBuilder
  $effect(() => {
    if (currentFullscreenNode?.nodeType === 'agent') {
      const graphConfig = currentFullscreenNode.data?.graph_config;
      if (graphConfig && Object.keys(graphConfig.nodes || {}).length > 1) {
        const agentId = currentFullscreenNode.data?.agentId;
        const agent = data.agents.find((a: Agent) => a.id === agentId);
        fullscreenNode.close();
        if (agent) {
          builderMode.setAgent();
          tick().then(() => agentBuilderRef.loadAgent(agent));
        }
      }
    }
  });

  // CodeMirror editor for Create Tool modal
  let createToolEditorContainer: HTMLDivElement = $state(undefined as any);
  let createToolEditorView: EditorView | null = null;

  function initCreateToolEditor() {
    destroyCreateToolEditor();

    if (!createToolEditorContainer) return;

    const startState = EditorState.create({
      doc: newToolCode,
      extensions: [
        basicSetup,
        python(),
        oneDark,
        EditorView.lineWrapping,
        EditorView.updateListener.of((update) => {
          if (update.docChanged) {
            newToolCode = update.state.doc.toString();
          }
        })
      ]
    });

    createToolEditorView = new EditorView({
      state: startState,
      parent: createToolEditorContainer
    });
  }

  function destroyCreateToolEditor() {
    if (createToolEditorView) {
      createToolEditorView.destroy();
      createToolEditorView = null;
    }
  }

  // Initialize editor when modal opens
  $effect(() => {
    if (showCreateToolModal && createToolEditorContainer) {
      untrack(() => initCreateToolEditor());
    }
  });

  // Cleanup on component destroy
  onDestroy(() => {
    destroyCreateToolEditor();
  });

  const nodeTypes = {
    selectorNode: ColorSelectorNode,
    toolNode: ToolNode,
    triggerNode: TriggerNode
  };

  const edgeTypes = {
    floating: FloatingEdge
  };

  const defaultEdgeOptions = {
    style: 'stroke-width: 3; stroke: black;',
    type: 'floating',
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: 'black'
    }
  };

  const connectionLineStyle = 'stroke: black; stroke-width: 3;';

  // Start with an empty canvas
  let nodes: Node[] = $state([]);

  let edges: Edge[] = $state([]);

  // Callback to update a node's data after tool is updated
  function handleToolUpdated(nodeId: string, updatedData: any) {
    const nodeIndex = nodes.findIndex(n => n.id === nodeId);
    if (nodeIndex !== -1) {
      nodes[nodeIndex] = {
        ...nodes[nodeIndex],
        data: {
          ...nodes[nodeIndex].data,
          ...updatedData
        }
      };
      // Trigger reactivity by reassigning the array
      nodes = [...nodes];
    }
  }

  function handleAgentUpdated(agentId: number, updatedAgent: Agent) {
    // Update sidebar data
    data.agents = data.agents.map(a => a.id === agentId ? { ...a, ...updatedAgent } : a);

    // Extract entry node config from graph_config
    const entryPoint = updatedAgent.graph_config?.entry_point || 'main';
    const entryNode = updatedAgent.graph_config?.nodes?.[entryPoint] || {};

    // Update any canvas nodes for this agent
    nodes = nodes.map(n => {
      if (n.data?.isAgent && n.data?.agentId === agentId) {
        return {
          ...n,
          data: {
            ...n.data,
            name: updatedAgent.name,
            description: updatedAgent.description || '',
            graph_config: updatedAgent.graph_config,
            system_prompt: entryNode.system_prompt || '',
            llm_provider: entryNode.llm_provider || '',
            tool_ids: entryNode.tool_ids || [],
          }
        };
      }
      return n;
    });
  }

  // Use tools and agents from database instead of hardcoded nodes
  let availableTools = $derived(data.tools.map(tool => tool.name));
  let availableAgents = $derived(data.agents.map(agent => agent.name));
  let availableFlows = $derived(data.flows);

  function addNode(nodeName: string, position: { x: number; y: number }) {
    // Check if it's an agent
    const agent = data.agents.find((a: Agent) => a.name === nodeName);
    if (agent) {
      // Extract entry node config from graph_config
      const entryPoint = agent.graph_config?.entry_point || 'main';
      const entryNode = agent.graph_config?.nodes?.[entryPoint] || {};

      const newNode: Node = {
        id: String(Date.now()),
        type: 'toolNode',
        data: {
          label: nodeName,
          handles: ['a'],
          isAgent: true,
          agentId: agent.id,
          name: agent.name,
          description: agent.description || '',
          graph_config: agent.graph_config,
          system_prompt: entryNode.system_prompt || '',
          llm_provider: entryNode.llm_provider || '',
          tool_ids: entryNode.tool_ids || [],
          output_paths: entryNode.output_paths || undefined,
          script_code: '',
          main_function: '',
          input_schema: null,
          output_schema: agent.output_schema || null,
          runtimeLLM: null
        },
        position,
        sourcePosition: Position.Right,
        targetPosition: Position.Left
      };
      nodes = [...nodes, newNode];
      return;
    }

    // Find the tool from the database
    const tool = data.tools.find((t: Tool) => t.name === nodeName);

    const newNode: Node = {
      id: String(Date.now()),
      type: nodeName === 'Color Picker' ? 'selectorNode' : 'toolNode',
      data: {
        label: nodeName,
        handles: ['a'],
        toolId: tool?.id, // Store tool ID for validation
        // Pass full tool data for ToolNode
        name: tool?.name || nodeName,
        description: tool?.description || '',
        script_code: tool?.script_code || '',
        main_function: tool?.main_function || '',
        input_schema: tool?.input_schema || null,
        output_schema: tool?.output_schema || null,
        runtimeLLM: null  // No LLM attached by default
      },
      position,
      sourcePosition: Position.Right,
      targetPosition: Position.Left
    };

    nodes = [...nodes, newNode];
  }

  function addTriggerNode(triggerType: string, position: { x: number; y: number }) {
    // Only allow one trigger node per flow
    const existingTrigger = nodes.find(n => n.type === 'triggerNode');
    if (existingTrigger) {
      validationSuccess = false;
      validationMessage = 'Only one trigger node allowed per flow.';
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 3000);
      return;
    }

    const newNode: Node = {
      id: String(Date.now()),
      type: 'triggerNode',
      data: {
        label: 'Text Input',
        name: 'Text Input',
        triggerType: triggerType,
        triggerValue: ''
      },
      position,
      sourcePosition: Position.Right,
      targetPosition: Position.Left
    };
    nodes = [...nodes, newNode];
  }

  async function onConnect(params) {
    try {
      console.log('Connecting:', params);

      // Get source and target nodes
      const sourceNode = nodes.find(n => n.id === params.source);
      const targetNode = nodes.find(n => n.id === params.target);

      // If both nodes have tool IDs (and neither is an agent), validate compatibility
      if (sourceNode?.data?.toolId && targetNode?.data?.toolId && !sourceNode?.data?.isAgent && !targetNode?.data?.isAgent) {
        // Use granular field-level validation
        const validation = await validateConnection(
          sourceNode.data.toolId,
          targetNode.data.toolId,
          params.sourceHandle || "",  // Empty string for whole output
          params.targetHandle || ""   // Empty string for whole input
        );

        if (!validation.compatible) {
          // Show error message with tool and field names
          const sourceName = sourceNode.data.name;
          const targetName = targetNode.data.name;
          const sourceFieldDesc = validation.source_field || 'output';
          const targetFieldDesc = validation.target_field || 'input';

          validationSuccess = false;
          validationMessage = `Invalid connection: ${sourceName}.${sourceFieldDesc} (${validation.source_type}) → ${targetName}.${targetFieldDesc} (${validation.target_type})`;
          showValidationToast = true;
          // Don't auto-dismiss validation errors - stay visible until user acts
          return; // Don't create the connection
        }
        // No message shown for valid connections
      }

      // Create the edge
      edges = addEdge(params, edges);
      console.log('Edges after connection:', edges);
    } catch (error) {
      console.error('Error creating edge:', error);
      validationSuccess = false;
      validationMessage = `Connection error: ${error}`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 5000);
    }
  }

  /**
   * Save the current visual flow to database
   */
  async function saveFlow() {
    try {
      isSaving = true;

      // Convert visual flow to graph_config
      const graphConfig = buildEnhancedGraphConfig(nodes, edges, data.tools);

      if (currentFlowId) {
        // Update existing flow - only update graph_config and conda_env
        // Name and description are not changed unless specifically updated through dialog
        const flowData: FlowUpdateRequest = {
          graph_config: graphConfig,
          conda_env: selectedCondaEnv || undefined
        };

        const updatedFlow = await updateFlow(currentFlowId, flowData);

        // Show success
        validationSuccess = true;
        validationMessage = `Flow "${updatedFlow.name}" updated successfully!`;
        showValidationToast = true;
        setTimeout(() => { showValidationToast = false; }, 3000);

        // Reset dialog state (if it was open)
        showSaveDialog = false;
        flowName = '';
        flowDescription = '';

      } else {
        // Create new flow
        const flowData: FlowCreateRequest = {
          name: flowName,
          description: flowDescription,
          graph_config: graphConfig,
          is_public: false,
          user_id: data.user?.id || 1,  // Use user ID or default to 1
          conda_env: selectedCondaEnv || undefined  // Store conda env as separate field
        };

        const createdFlow = await createFlow(flowData);

        // Store the current flow ID
        currentFlowId = createdFlow.id;

        // Show success
        validationSuccess = true;
        validationMessage = `Flow "${createdFlow.name}" created successfully!`;
        showValidationToast = true;
        setTimeout(() => { showValidationToast = false; }, 3000);

        // Reset and close dialog
        showSaveDialog = false;
        flowName = '';
        flowDescription = '';
      }

    } catch (error) {
      // Show error
      validationSuccess = false;
      validationMessage = `Failed to save flow: ${error}`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 5000);
    } finally {
      isSaving = false;
    }
  }

  /**
   * Load a saved flow and recreate its nodes and edges
   */
  async function loadFlow(flowId: number) {
    try {
      // Fetch flow details with graph_config
      const flow = await getFlowDetails(flowId);

      if (!flow.graph_config) {
        throw new Error('Flow has no graph_config');
      }

      const graphConfig = flow.graph_config;

      // Clear existing nodes and edges
      nodes = [];
      edges = [];

      // Recreate nodes from graph_config
      const nodeMap = new Map<string, Node>();

      for (const [nodeId, nodeConfig] of Object.entries(graphConfig.nodes)) {
        if (nodeConfig.node_type === 'tool') {
          // Find the tool from data.tools
          const tool = data.tools.find((t: Tool) => t.id === nodeConfig.id);

          if (tool) {
            // Restore LLM configuration by looking up the model name
            let runtimeLLM = null;
            if (nodeConfig.model_name) {
              // Find the LLM provider by name from the loaded providers
              runtimeLLM = llmProviders.find(p => p.name === nodeConfig.model_name) || null;
            }

            const newNode: Node = {
              id: nodeId,
              type: 'toolNode',
              data: {
                label: tool.name,
                handles: ['a'],
                toolId: tool.id,
                name: tool.name,
                description: tool.description,
                script_code: tool.script_code,
                main_function: tool.main_function || '',
                input_schema: tool.input_schema,
                output_schema: tool.output_schema,
                runtimeLLM: runtimeLLM,
                parameterValues: nodeConfig.input_values || {}
              },
              position: { x: 100 + Math.random() * 400, y: 100 + Math.random() * 300 },
              sourcePosition: Position.Right,
              targetPosition: Position.Left
            };
            nodeMap.set(nodeId, newNode);
          }
        } else if (nodeConfig.node_type === 'agent') {
          // Find the agent from data.agents
          const agent = data.agents.find((a: Agent) => a.id === nodeConfig.id);

          if (agent) {
            const entryPoint = agent.graph_config?.entry_point || 'main';
            const entryNodeConfig = agent.graph_config?.nodes?.[entryPoint] || {};

            const newNode: Node = {
              id: nodeId,
              type: 'toolNode',
              data: {
                label: agent.name,
                handles: ['a'],
                isAgent: true,
                agentId: agent.id,
                name: agent.name,
                description: agent.description || '',
                graph_config: agent.graph_config,
                system_prompt: entryNodeConfig.system_prompt || '',
                llm_provider: entryNodeConfig.llm_provider || '',
                tool_ids: entryNodeConfig.tool_ids || [],
                eval_ids: entryNodeConfig.eval_ids || [],
                output_paths: entryNodeConfig.output_paths || undefined,
                script_code: '',
                main_function: '',
                input_schema: null,
                output_schema: agent.output_schema || null,
                runtimeLLM: entryNodeConfig.llm_provider
                  ? llmProviders.find(p => p.name === entryNodeConfig.llm_provider) || null
                  : null
              },
              position: { x: 100 + Math.random() * 400, y: 100 + Math.random() * 300 },
              sourcePosition: Position.Right,
              targetPosition: Position.Left
            };
            nodeMap.set(nodeId, newNode);
          }
        } else if (nodeConfig.node_type === 'trigger') {
          const newNode: Node = {
            id: nodeId,
            type: 'triggerNode',
            data: {
              label: 'Text Input',
              name: 'Text Input',
              triggerType: 'text_input',
              triggerValue: nodeConfig.input_value || ''
            },
            position: { x: 50, y: 200 },
            sourcePosition: Position.Right,
            targetPosition: Position.Left
          };
          nodeMap.set(nodeId, newNode);
        }
      }

      // Recreate edges from graph_config
      const newEdges: Edge[] = [];
      for (const edgeConfig of graphConfig.edges) {
        if (nodeMap.has(edgeConfig.from_node) && nodeMap.has(edgeConfig.to_node)) {
          const edge: Edge = {
            id: `${edgeConfig.from_node}-${edgeConfig.to_node}`,
            source: edgeConfig.from_node,
            target: edgeConfig.to_node,
            ...defaultEdgeOptions
          };

          // Note: We don't restore sourceHandle/targetHandle here
          // The mapping is stored in graph_config and used by the backend
          // Visual edges don't need handle information when loading

          newEdges.push(edge);
        }
      }

      // Auto-layout the nodes using ELK
      const layoutedNodes = await autoLayoutNodes(Array.from(nodeMap.values()), newEdges);
      nodes = layoutedNodes;

      // Now restore sourceHandle/targetHandle from mappings (after layout to avoid ELK errors)
      for (const edgeConfig of graphConfig.edges) {
        const edge = newEdges.find(e => e.source === edgeConfig.from_node && e.target === edgeConfig.to_node);
        if (edge && edgeConfig.mapping && Object.keys(edgeConfig.mapping).length > 0) {
          const [outputField, inputParam] = Object.entries(edgeConfig.mapping)[0];
          edge.sourceHandle = outputField;
          edge.targetHandle = inputParam;
        }
      }

      edges = newEdges;

      // Set the conda environment from the loaded flow
      selectedCondaEnv = flow.conda_env;

      // Store the current flow ID
      currentFlowId = flowId;

      // Show success
      validationSuccess = true;
      validationMessage = `Flow "${flow.name}" loaded successfully!`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 3000);

    } catch (error) {
      validationSuccess = false;
      validationMessage = `Failed to load flow: ${error}`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 5000);
    }
  }

  /**
   * Get initial input values from the entry point node
   */
  function getEntryPointInputValues(): Record<string, any> {
    // If there's a trigger node, use its text value as input
    const triggerNode = nodes.find(n => n.type === 'triggerNode');
    if (triggerNode && triggerNode.data.triggerValue) {
      return { text: triggerNode.data.triggerValue };
    }

    // Fallback: find the entry point node (node with no incoming edges)
    const nodesWithIncoming = new Set(edges.map(e => e.target));
    const entryNode = nodes.find(n => n.type === 'toolNode' && !nodesWithIncoming.has(n.id));

    if (entryNode && entryNode.data.parameterValues) {
      return entryNode.data.parameterValues;
    }

    return {};
  }

  /**
   * Execute a saved flow
   */
  async function runFlow(flowId: number, initialInput: Record<string, any>) {
    try {
      const agentLlms: Record<string, string> = {};
      for (const node of nodes) {
        if (node.data.isAgent && node.data.llm_provider) {
          agentLlms[node.id] = node.data.llm_provider;
        }
      }
      const userId = data.user?.id || 1;
      const result = await executeFlow(flowId, initialInput, selectedCondaEnv, userId, agentLlms);

      // Capture execution ID and open info panel
      if (result.execution_id) {
        lastExecutionId = result.execution_id;
        showInfoPanel = true;
      }

      if (result.status === 'completed') {
        console.log('✓ Flow completed successfully');

        // Auto-run evaluations if enabled
        if (evalsEnabled && result.execution_id) {
          runPostExecutionEvals(result.execution_id);
        }

        // Show success
        validationSuccess = true;
        validationMessage = `Flow completed! Output: ${JSON.stringify(result.final_output)}`;
        showValidationToast = true;
        setTimeout(() => { showValidationToast = false; }, 5000);
      } else {
        console.error('✗ Flow failed:', result.error);
        validationSuccess = false;
        validationMessage = `Flow failed: ${result.error}`;
        showValidationToast = true;
        setTimeout(() => { showValidationToast = false; }, 5000);
      }

    } catch (error) {
      console.error('Flow execution error:', error);
      validationSuccess = false;
      validationMessage = `Failed to execute flow: ${error}`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 5000);
    }
  }

  /**
   * Run evaluations for agent nodes after flow execution completes.
   * Looks up eval_ids from each agent's graph_config and fires evaluateExecution calls.
   */
  async function runPostExecutionEvals(executionId: number) {
    evalsRunning = true;
    try {
      const execution = await fetchExecution(executionId);
      if (!execution || !execution.children) return;

      const userId = data.user?.id || 1;
      const evalPromises: Promise<any>[] = [];

      for (const child of execution.children) {
        if (child.execution_type !== 'agent' || !child.langfuse_trace_id) continue;

        // Find the agent on the canvas to get its graph_config with eval_ids
        const agentNode = nodes.find(n => n.data?.isAgent && n.data?.name === child.name);
        if (!agentNode) continue;

        // Per-node evals from the entry point
        const graphConfig = agentNode.data.graph_config;
        const entryPoint = graphConfig?.entry_point || 'main';
        const nodeConfig = graphConfig?.nodes?.[entryPoint] || agentNode.data;
        const evalIds: number[] = nodeConfig.eval_ids || agentNode.data.eval_ids || [];

        // Top-level complex agent evals
        const topLevelEvalIds: number[] = graphConfig?.eval_ids || [];
        const allEvalIds = [...new Set([...evalIds, ...topLevelEvalIds])];

        if (allEvalIds.length > 0) {
          // Use the agent's LLM provider for the judge
          const agentLlmProvider = nodeConfig.llm_provider || agentNode.data.llm_provider || '';
          if (agentLlmProvider) {
            evalPromises.push(evaluateExecution(child.id, userId, allEvalIds, agentLlmProvider));
          }
        }
      }

      if (evalPromises.length > 0) {
        await Promise.all(evalPromises);
      }
    } catch (error) {
      console.error('Post-execution evals failed:', error);
    } finally {
      evalsRunning = false;
    }
  }

  /**
   * Delete a flow
   */
  async function handleDeleteFlow(flowId: number, flowName: string, event: Event) {
    // Stop propagation so the click doesn't trigger loadFlow
    event.stopPropagation();

    // Confirm before deleting
    if (!confirm(`Are you sure you want to delete "${flowName}"?`)) {
      return;
    }

    try {
      await deleteFlow(flowId);

      // Remove from the available flows list
      data.flows = data.flows.filter(f => f.id !== flowId);

      // Clear the current flow if it's the one being deleted
      if (currentFlowId === flowId) {
        currentFlowId = null;
        nodes = [];
        edges = [];
      }

      // Show success
      validationSuccess = true;
      validationMessage = `Flow "${flowName}" deleted successfully!`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 3000);

    } catch (error) {
      validationSuccess = false;
      validationMessage = `Failed to delete flow: ${error}`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 5000);
    }
  }

  async function handleDeleteAgent(agentId: number, agentName: string, event: Event) {
    event.stopPropagation();
    if (!confirm(`Are you sure you want to delete "${agentName}"?`)) return;

    try {
      await deleteAgent(agentId);
      data.agents = data.agents.filter(a => a.id !== agentId);
      validationSuccess = true;
      validationMessage = `Agent "${agentName}" deleted successfully!`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 3000);
    } catch (error) {
      validationSuccess = false;
      validationMessage = `Failed to delete agent: ${error}`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 5000);
    }
  }

  async function handleDeleteTool(toolId: number, toolName: string, event: Event) {
    event.stopPropagation();
    if (!confirm(`Are you sure you want to delete "${toolName}"?`)) return;

    try {
      await deleteTool(toolId);
      data.tools = data.tools.filter(t => t.id !== toolId);
      validationSuccess = true;
      validationMessage = `Tool "${toolName}" deleted successfully!`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 3000);
    } catch (error) {
      validationSuccess = false;
      validationMessage = `Failed to delete tool: ${error}`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 5000);
    }
  }

  /**
   * Create a new tool
   */
  async function handleCreateTool() {
    if (!newToolName.trim()) {
      validationSuccess = false;
      validationMessage = 'Please enter a tool name';
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 3000);
      return;
    }

    if (!newToolMainFunction.trim()) {
      validationSuccess = false;
      validationMessage = 'Please enter a main function name';
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 3000);
      return;
    }

    if (!newToolCode.trim()) {
      validationSuccess = false;
      validationMessage = 'Please enter Python script code for the tool';
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 3000);
      return;
    }

    try {
      isCreatingTool = true;

      const userId = data.user?.id || 1;
      const createdTool = await createPythonScriptTool(userId, {
        name: newToolName.trim(),
        description: newToolDescription.trim(),
        script_code: newToolCode,
        main_function: newToolMainFunction.trim(),
        is_public: newToolIsPublic
      });

      // Add the new tool to the available tools list
      data.tools = [...data.tools, createdTool];

      // Show success
      validationSuccess = true;
      validationMessage = `Tool "${createdTool.name}" created successfully!`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 3000);

      // Reset and close modal
      closeCreateToolModal();
      newToolName = '';
      newToolDescription = '';
      newToolCode = '';
      newToolMainFunction = '';
      newToolIsPublic = false;
      showWriteWithAI = false;
      additionalInstructions = '';
      showEditWithAI = false;
      editingInstructions = '';

    } catch (error) {
      validationSuccess = false;
      validationMessage = `Failed to create tool: ${error}`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 5000);
    } finally {
      isCreatingTool = false;
    }
  }

  /**
   * Generate tool code using AI with streaming
   */
  async function handleGenerateWithAI() {
    // Validate name and description
    if (!newToolName.trim()) {
      validationSuccess = false;
      validationMessage = 'Please enter a tool name before generating code';
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 3000);
      return;
    }
    if (!newToolDescription.trim()) {
      validationSuccess = false;
      validationMessage = 'Please enter a tool description before generating code';
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 3000);
      return;
    }

    // Validate LLM provider is selected
    if (!selectedLLMProvider) {
      validationSuccess = false;
      validationMessage = 'Please select an LLM provider from the "Attach LLM" panel';
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 5000);
      return;
    }

    try {
      isGeneratingCode = true;

      // Clear existing code
      newToolCode = '';
      if (createToolEditorView) {
        const transaction = createToolEditorView.state.update({
          changes: {
            from: 0,
            to: createToolEditorView.state.doc.length,
            insert: ''
          }
        });
        createToolEditorView.dispatch(transaction);
      }

      // Start streaming
      await generateToolCodeStream(
        {
          tool_name: newToolName.trim(),
          tool_description: newToolDescription.trim(),
          model: selectedLLMProvider.name,
          additional_instructions: additionalInstructions.trim() || undefined,
          user_id: data.user?.id,
        },
        // onChunk: append text to editor as it arrives
        (chunk: string) => {
          newToolCode += chunk;
          if (createToolEditorView) {
            const transaction = createToolEditorView.state.update({
              changes: {
                from: createToolEditorView.state.doc.length,
                to: createToolEditorView.state.doc.length,
                insert: chunk
              }
            });
            createToolEditorView.dispatch(transaction);
          }
        },
        // onDone: update with final cleaned code and main function
        (scriptCode: string, mainFunction: string) => {
          // Replace editor content with cleaned code (markdown stripped)
          newToolCode = scriptCode;
          if (createToolEditorView) {
            const transaction = createToolEditorView.state.update({
              changes: {
                from: 0,
                to: createToolEditorView.state.doc.length,
                insert: scriptCode
              }
            });
            createToolEditorView.dispatch(transaction);
          }

          // Update main function name
          newToolMainFunction = mainFunction;

          // Show success toast
          validationSuccess = true;
          validationMessage = 'Code generated successfully! Review and edit as needed.';
          showValidationToast = true;
          setTimeout(() => { showValidationToast = false; }, 3000);

          isGeneratingCode = false;
        },
        // onError: show error message
        (error: string) => {
          validationSuccess = false;
          validationMessage = `Failed to generate code: ${error}`;
          showValidationToast = true;
          setTimeout(() => { showValidationToast = false; }, 5000);
          isGeneratingCode = false;
        }
      );

    } catch (error) {
      validationSuccess = false;
      validationMessage = `Failed to generate code: ${error}`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 5000);
      isGeneratingCode = false;
    }
  }

  /**
   * Edit existing code using AI with streaming
   */
  async function handleEditWithAI() {
    // Validate editing instructions
    if (!editingInstructions.trim()) {
      validationSuccess = false;
      validationMessage = 'Please enter editing instructions';
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 3000);
      return;
    }

    // Validate code exists
    if (!newToolCode.trim()) {
      validationSuccess = false;
      validationMessage = 'No code to edit. Please write or generate code first.';
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 3000);
      return;
    }

    // Validate LLM provider is selected
    if (!selectedLLMProvider) {
      validationSuccess = false;
      validationMessage = 'Please select an LLM provider';
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 5000);
      return;
    }

    try {
      isEditingCode = true;

      // Save the existing code before clearing
      const existingCode = newToolCode;

      // Clear existing code
      newToolCode = '';
      if (createToolEditorView) {
        const transaction = createToolEditorView.state.update({
          changes: {
            from: 0,
            to: createToolEditorView.state.doc.length,
            insert: ''
          }
        });
        createToolEditorView.dispatch(transaction);
      }

      // Start streaming edited code
      await editToolCodeStream(
        {
          existing_code: existingCode,
          editing_instructions: editingInstructions.trim(),
          tool_name: newToolName.trim() || 'tool',
          tool_description: newToolDescription.trim() || '',
          model: selectedLLMProvider.name,
          user_id: data.user?.id,
        },
        // onChunk: append text to editor as it arrives
        (chunk: string) => {
          newToolCode += chunk;
          if (createToolEditorView) {
            const transaction = createToolEditorView.state.update({
              changes: {
                from: createToolEditorView.state.doc.length,
                to: createToolEditorView.state.doc.length,
                insert: chunk
              }
            });
            createToolEditorView.dispatch(transaction);
          }
        },
        // onDone: update with final cleaned code and main function
        (scriptCode: string, mainFunction: string) => {
          // Replace editor content with cleaned code (markdown stripped)
          newToolCode = scriptCode;
          if (createToolEditorView) {
            const transaction = createToolEditorView.state.update({
              changes: {
                from: 0,
                to: createToolEditorView.state.doc.length,
                insert: scriptCode
              }
            });
            createToolEditorView.dispatch(transaction);
          }

          // Update main function name if provided
          if (mainFunction) {
            newToolMainFunction = mainFunction;
          }

          // Show success toast
          validationSuccess = true;
          validationMessage = 'Code edited successfully! Review the changes.';
          showValidationToast = true;
          setTimeout(() => { showValidationToast = false; }, 3000);

          isEditingCode = false;
        },
        // onError: show error message
        (error: string) => {
          validationSuccess = false;
          validationMessage = `Failed to edit code: ${error}`;
          showValidationToast = true;
          setTimeout(() => { showValidationToast = false; }, 5000);
          isEditingCode = false;
        }
      );

    } catch (error) {
      validationSuccess = false;
      validationMessage = `Failed to edit code: ${error}`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 5000);
      isEditingCode = false;
    }
  }

  function closeCreateToolModal() {
    destroyCreateToolEditor();
    showCreateToolModal = false;
  }

  /**
   * Create a new agent
   */
  async function handleCreateAgent() {
    if (!newAgentName.trim()) {
      validationSuccess = false;
      validationMessage = 'Please enter an agent name';
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 3000);
      return;
    }

    if (!newAgentSystemPrompt.trim()) {
      validationSuccess = false;
      validationMessage = 'Please enter a system prompt for the agent';
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 3000);
      return;
    }

    if (!newAgentUserPrompt.includes('{input}')) {
      validationSuccess = false;
      validationMessage = 'User prompt must contain {input} so the agent receives runtime input';
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 5000);
      return;
    }

    try {
      isCreatingAgent = true;

      const userId = data.user?.id || 1;
      const agentData: AgentCreateData = {
        name: newAgentName.trim(),
        description: newAgentDescription.trim(),
        graph_config: {
          nodes: {
            "main": {
              agent_type: "pydanticai",
              name: newAgentName.trim(),
              description: newAgentDescription.trim(),
              system_prompt: newAgentSystemPrompt.trim(),
              user_prompt: newAgentUserPrompt.trim(),
              llm_provider: newAgentLLMProvider || '',
              tool_ids: newAgentSelectedTools,
              eval_ids: newAgentSelectedEvals,
              ...(newAgentOutputPaths.filter(p => p.name.trim()).length > 0 ? {
                output_paths: Object.fromEntries(
                  newAgentOutputPaths
                    .filter(p => p.name.trim())
                    .map(p => [p.name.trim(), {
                      description: p.description.trim(),
                      return_behavior: p.return_behavior
                    }])
                )
              } : {})
            }
          },
          edges: [],
          entry_point: "main",
          exit_points: ["main"]
        }
      };

      const createdAgent = await createAgent(userId, agentData);

      // Add the new agent to the available agents list
      data.agents = [...data.agents, createdAgent];

      // Show success
      validationSuccess = true;
      validationMessage = `Agent "${createdAgent.name}" created successfully!`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 3000);

      // Reset and close modal
      closeCreateAgentModal();

    } catch (error) {
      validationSuccess = false;
      validationMessage = `Failed to create agent: ${error}`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 5000);
    } finally {
      isCreatingAgent = false;
    }
  }

  /**
   * Generate system prompt using AI with streaming
   */
  async function handleGeneratePrompt() {
    if (!newAgentName.trim()) {
      validationSuccess = false;
      validationMessage = 'Please enter an agent name before generating a prompt';
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 3000);
      return;
    }
    if (!newAgentDescription.trim()) {
      validationSuccess = false;
      validationMessage = 'Please enter an agent description before generating a prompt';
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 3000);
      return;
    }
    if (!selectedLLMProvider) {
      validationSuccess = false;
      validationMessage = 'Please select an LLM provider';
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 5000);
      return;
    }

    try {
      isGeneratingPrompt = true;
      newAgentSystemPrompt = '';
      newAgentUserPrompt = '';

      // Get selected tool names
      const selectedToolNames = data.tools
        .filter((t: Tool) => newAgentSelectedTools.includes(t.id))
        .map((t: Tool) => t.name);

      const requestData = {
        agent_name: newAgentName.trim(),
        agent_description: newAgentDescription.trim(),
        tool_names: selectedToolNames,
        model: selectedLLMProvider.name,
        additional_instructions: promptAdditionalInstructions.trim() || undefined,
        user_id: data.user?.id,
      };

      // Pass 1: Generate system prompt
      let generatedSystemPrompt = '';

      await generateSystemPromptStream(
        requestData,
        (chunk: string) => {
          newAgentSystemPrompt += chunk;
        },
        (systemPrompt: string) => {
          newAgentSystemPrompt = systemPrompt;
          generatedSystemPrompt = systemPrompt;
        },
        (error: string) => {
          validationSuccess = false;
          validationMessage = `Failed to generate system prompt: ${error}`;
          showValidationToast = true;
          setTimeout(() => { showValidationToast = false; }, 5000);
          isGeneratingPrompt = false;
        }
      );

      // Pass 2: Generate user prompt (aware of the system prompt)
      if (generatedSystemPrompt) {
        await generateUserPromptStream(
          { ...requestData, generated_system_prompt: generatedSystemPrompt },
          (chunk: string) => {
            newAgentUserPrompt += chunk;
          },
          (userPrompt: string) => {
            newAgentUserPrompt = userPrompt;
            validationSuccess = true;
            validationMessage = 'System and user prompts generated successfully!';
            showValidationToast = true;
            setTimeout(() => { showValidationToast = false; }, 3000);
            isGeneratingPrompt = false;
          },
          (error: string) => {
            validationSuccess = false;
            validationMessage = `Failed to generate user prompt: ${error}`;
            showValidationToast = true;
            setTimeout(() => { showValidationToast = false; }, 5000);
            isGeneratingPrompt = false;
          }
        );
      }
    } catch (error) {
      validationSuccess = false;
      validationMessage = `Failed to generate prompt: ${error}`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 5000);
      isGeneratingPrompt = false;
    }
  }

  function toggleToolSelection(toolId: number) {
    if (newAgentSelectedTools.includes(toolId)) {
      newAgentSelectedTools = newAgentSelectedTools.filter(id => id !== toolId);
    } else {
      newAgentSelectedTools = [...newAgentSelectedTools, toolId];
    }
  }

  function toggleEvalSelection(evalId: number) {
    if (newAgentSelectedEvals.includes(evalId)) {
      newAgentSelectedEvals = newAgentSelectedEvals.filter(id => id !== evalId);
    } else {
      newAgentSelectedEvals = [...newAgentSelectedEvals, evalId];
    }
  }

  function addAgentOutputPath() {
    newAgentOutputPaths = [...newAgentOutputPaths, { name: '', description: '', return_behavior: 'node_output' }];
  }

  function removeAgentOutputPath(index: number) {
    newAgentOutputPaths = newAgentOutputPaths.filter((_, i) => i !== index);
  }

  function openAgentDetails(agent: Agent) {
    // Complex agent: load into AgentBuilder canvas
    const nodeCount = Object.keys(agent.graph_config?.nodes || {}).length;
    if (nodeCount > 1) {
      builderMode.setAgent();
      tick().then(() => {
        agentBuilderRef.loadAgent(agent);
      });
      return;
    }

    const entryPoint = agent.graph_config?.entry_point || 'main';
    const entryNode = agent.graph_config?.nodes?.[entryPoint] || {};

    fullscreenNode.open({
      nodeId: `sidebar-agent-${agent.id}`,
      nodeType: 'agent',
      data: {
        name: agent.name,
        description: agent.description || '',
        agentId: agent.id,
        graph_config: agent.graph_config,
        system_prompt: entryNode.system_prompt || '',
        llm_provider: entryNode.llm_provider || '',
        tool_ids: entryNode.tool_ids || [],
        eval_ids: entryNode.eval_ids || [],
        output_schema: agent.output_schema || null
      }
    });
  }

  function closeCreateAgentModal() {
    showCreateAgentModal = false;
    newAgentName = '';
    newAgentDescription = '';
    newAgentSystemPrompt = '';
    newAgentUserPrompt = '{input}';
    newAgentSelectedTools = [];
    newAgentSelectedEvals = [];
    newAgentLLMProvider = '';
    showGeneratePromptAI = false;
    promptAdditionalInstructions = '';
    isGeneratingPrompt = false;
    newAgentOutputPaths = [];
    isConfiguringComplexNode = false;
    pendingAgentTemplate = null;
  }

  function handleNodeDataUpdated(nodeId: string, updatedData: any) {
    // Update agent builder canvas nodes (for unsaved nodes edited via modal)
    if (currentMode === 'agent' && agentBuilderRef) {
      agentBuilderRef.updateNodeData(nodeId, updatedData);
    }
    // Update flow canvas nodes
    nodes = nodes.map(n =>
      n.id === nodeId ? { ...n, data: { ...n.data, ...updatedData } } : n
    );
  }

  function handleAgentUpdatedFromBuilder(event: CustomEvent<Agent>) {
    data.agents = data.agents.map(a => a.id === event.detail.id ? event.detail : a);
  }

  // Agent Builder handlers
  function enterAgentBuilderMode() {
    builderMode.setAgent();
  }

  function handleAgentCreated(event: CustomEvent<Agent>) {
    data.agents = [...data.agents, event.detail];
  }

  function handleAgentBuilderBack() {
    builderMode.setFlow();
  }

  function handleConfigureNewAgent(event: CustomEvent<{ template: AgentTemplate }>) {
    const { template } = event.detail;
    pendingAgentTemplate = template;
    isConfiguringComplexNode = true;
    newAgentName = template.name;
    newAgentDescription = template.description;
    newAgentSystemPrompt = template.defaultSystemPrompt;
    newAgentUserPrompt = template.defaultUserPrompt;
    newAgentSelectedTools = [];
    newAgentSelectedEvals = [];
    newAgentLLMProvider = '';
    newAgentOutputPaths = [];
    showCreateAgentModal = true;
  }

  function handleAddNodeToCanvas() {
    if (!pendingAgentTemplate) return;

    if (!newAgentUserPrompt.includes('{input}')) {
      validationSuccess = false;
      validationMessage = 'User prompt must contain {input} so the agent receives runtime input';
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 5000);
      return;
    }

    const outputPaths = newAgentOutputPaths.filter(p => p.name.trim()).length > 0
      ? Object.fromEntries(
          newAgentOutputPaths
            .filter(p => p.name.trim())
            .map(p => [p.name.trim(), {
              description: p.description.trim(),
              return_behavior: p.return_behavior
            }])
        )
      : undefined;

    agentBuilderRef.addConfiguredAgentNode({
      name: newAgentName.trim(),
      description: newAgentDescription.trim(),
      system_prompt: newAgentSystemPrompt.trim(),
      user_prompt: newAgentUserPrompt.trim(),
      llm_provider: newAgentLLMProvider || '',
      tool_ids: newAgentSelectedTools,
      eval_ids: newAgentSelectedEvals,
      output_paths: outputPaths,
      agentType: pendingAgentTemplate.type
    });

    closeCreateAgentModal();
  }
</script>

<div class="app-container">
  <!-- Validation Toast -->
  {#if showValidationToast}
    <div class="validation-toast" class:success={validationSuccess} class:error={!validationSuccess}>
      <span class="toast-message">{validationMessage}</span>
      <button class="toast-close" onclick={() => showValidationToast = false}>×</button>
    </div>
  {/if}

  <div class="node-window">
    {#if currentMode === 'flow'}
      <!-- FLOW BUILDER SIDEBAR -->
      <div class="user-section">
        {#if data.user}
          <div class="current-user">
            <strong>{data.user.username}</strong>
            <small>{data.user.email}</small>
          </div>
          <form method="POST" action="/logout" class="mt-2">
            <Button type="submit" variant="outline" class="w-full" size="sm">
              Logout
            </Button>
          </form>
        {:else}
          <div class="space-y-2">
            <Button class="w-full" onclick={() => window.location.href = '/login'}>
              Sign In
            </Button>
            <Button variant="outline" class="w-full" onclick={() => window.location.href = '/register'}>
              Register
            </Button>
          </div>
        {/if}
      </div>

      <CondaEnvironmentsPanel bind:selectedEnv={selectedCondaEnv} />

      <LLMProvidersPanel bind:selectedProvider={selectedLLMProvider} bind:providers={llmProviders} userId={data.user?.id} />

      <EvaluationManager userId={data.user?.id || 1} onchange={(evals) => data.evaluations = evals} />

      <div class="triggers-section">
        <h4>Triggers</h4>
        <div
          class="draggable-node trigger-draggable"
          draggable="true"
          ondragstart={(event) => {
            event.dataTransfer?.setData('application/json', JSON.stringify({ itemType: 'trigger', triggerType: 'text_input' }));
            event.dataTransfer?.setData('text/plain', '__trigger__text_input');
          }}
        >
          Text Input
        </div>
      </div>

      <div class="flows-section">
        <h4>Available Flows</h4>
        <Button class="w-full mb-2" size="sm" onclick={() => showSaveDialog = true}>
          Create New Flow
        </Button>
        {#each availableFlows as flow}
          <div
            class="flow-item"
            onclick={() => loadFlow(flow.id)}
          >
            <span class="flow-item-name">{flow.name}</span>
            <button
              class="flow-item-delete"
              onclick={(e) => handleDeleteFlow(flow.id, flow.name, e)}
              title="Delete flow"
            >
              ×
            </button>
          </div>
        {/each}
      </div>

      <div class="tools-section">
        <h4>Available Tools</h4>
        <Button size="sm" onclick={() => showCreateToolModal = true} class="w-full mb-2 bg-blue-600 hover:bg-blue-700">
          {#snippet children()}
            Create New Tool
          {/snippet}
        </Button>
        {#each data.tools as tool}
          <div
            class="sidebar-item"
            draggable="true"
            ondragstart={(event) => event.dataTransfer?.setData('text/plain', tool.name)}
          >
            <span class="sidebar-item-name">{tool.name}</span>
            <button
              class="sidebar-item-delete"
              onclick={(e) => handleDeleteTool(tool.id, tool.name, e)}
              title="Delete tool"
            >
              ×
            </button>
          </div>
        {/each}
      </div>

      <div class="agents-section">
        <h4>Available Agents</h4>
        <div class="agent-buttons">
          <Button size="sm" onclick={() => showCreateAgentModal = true} class="w-full mb-1 bg-green-600 hover:bg-green-700">
            {#snippet children()}
              Create Simple Agent
            {/snippet}
          </Button>
          <Button size="sm" onclick={enterAgentBuilderMode} class="w-full mb-2 bg-purple-600 hover:bg-purple-700">
            {#snippet children()}
              Create Complex Agent
            {/snippet}
          </Button>
        </div>
        {#each data.agents as agent}
          <div
            class="sidebar-item"
            draggable="true"
            ondragstart={(event) => event.dataTransfer?.setData('text/plain', agent.name)}
            onclick={() => openAgentDetails(agent)}
            title="Click to view details, drag to add to canvas"
          >
            <span class="sidebar-item-name">
              {agent.name}
              {#if agent.graph_config && Object.keys(agent.graph_config.nodes || {}).length > 1}
                <span class="composed-badge">⚡</span>
              {/if}
            </span>
            <button
              class="sidebar-item-delete"
              onclick={(e) => handleDeleteAgent(agent.id, agent.name, e)}
              title="Delete agent"
            >
              ×
            </button>
          </div>
        {/each}
      </div>

    {/if}
  </div>

  {#if currentMode === 'flow'}
    <!-- FLOW BUILDER CANVAS -->
    <div
      class="flow-container"
      role="application"
      ondragover={(event) => event.preventDefault()}
      ondrop={(event) => {
        event.preventDefault();

        // Get the flow container bounds
        const flowContainer = event.currentTarget;
        const rect = flowContainer.getBoundingClientRect();

        // Convert screen coordinates to flow coordinates using viewport
        const position = {
          x: (event.clientX - rect.left - viewport.x) / viewport.zoom,
          y: (event.clientY - rect.top - viewport.y) / viewport.zoom
        };

        // Check for trigger node drop (uses application/json payload)
        const jsonData = event.dataTransfer?.getData('application/json');
        if (jsonData) {
          try {
            const parsed = JSON.parse(jsonData);
            if (parsed.itemType === 'trigger') {
              addTriggerNode(parsed.triggerType, position);
              return;
            }
          } catch (e) { /* not JSON, fall through */ }
        }

        const nodeName = event.dataTransfer?.getData('text/plain');
        if (!nodeName) return;
        addNode(nodeName, position);
      }}
    >
      <!-- Flow Controls -->
      <div class="flow-controls">
        <Button onclick={() => {
          if (currentFlowId) {
            // Update existing flow directly
            saveFlow();
          } else {
            // Show dialog for new flow
            showSaveDialog = true;
          }
        }}>
          {currentFlowId ? 'Update Flow' : 'Save Flow'}
        </Button>
        {#if currentFlowId}
          <Button
            variant="outline"
            onclick={() => {
              currentFlowId = null;
              nodes = [];
              edges = [];
              selectedCondaEnv = null;
            }}
            class="ml-2"
          >
            Clear
          </Button>
        {/if}
        <Button
          onclick={() => currentFlowId && runFlow(currentFlowId, getEntryPointInputValues())}
          disabled={!currentFlowId}
          class="ml-2 bg-green-600 hover:bg-green-700 text-white"
        >
          ▶ Run
        </Button>
        <label class="evals-toggle">
          <input type="checkbox" bind:checked={evalsEnabled} />
          <span>Evals</span>
        </label>
      </div>

      <div class="canvas-area">
        <SvelteFlow
          bind:nodes
          {nodeTypes}
          bind:edges
          {edgeTypes}
          {defaultEdgeOptions}
          connectionLineType={ConnectionLineType.Straight}
          {connectionLineStyle}
          style="background: #1A192B"
          fitView
          bind:viewport
          onconnect={onConnect}
        >
          <Background />
          <Controls />
          <MiniMap />
        </SvelteFlow>

        <!-- Info panel toggle button (only shown when panel is closed) -->
        {#if !showInfoPanel}
          <button
            class="info-toggle-btn"
            onclick={() => showInfoPanel = true}
            title="Show Info Panel"
          >
            &#9432; Info
          </button>
        {/if}
      </div>

      {#if showInfoPanel}
        <InfoPanel
          executionId={lastExecutionId}
          flowName={flowName}
          userId={data.user?.id || 1}
          evalsEnabled={evalsEnabled}
          {evalsRunning}
          onclose={() => showInfoPanel = false}
        />
      {/if}
    </div>

  {:else}
    <!-- AGENT BUILDER (Self-contained component) -->
    <AgentBuilder
      bind:this={agentBuilderRef}
      agents={data.agents}
      evaluations={data.evaluations}
      userId={data.user?.id || 1}
      on:back={handleAgentBuilderBack}
      on:agentCreated={handleAgentCreated}
      on:agentUpdated={handleAgentUpdatedFromBuilder}
      on:configureNewAgent={handleConfigureNewAgent}
    />
  {/if}

  <!-- Save Flow Dialog -->
  {#if showSaveDialog}
    <div class="dialog-overlay" onclick={() => showSaveDialog = false}>
      <div class="dialog-content" onclick={(e) => e.stopPropagation()}>
        <div class="dialog-header">
          <h3>Save Flow</h3>
          <button class="dialog-close" onclick={() => showSaveDialog = false}>×</button>
        </div>

        <div class="dialog-body">
          <div class="form-field">
            <Label for="flowName">Flow Name</Label>
            <Input id="flowName" bind:value={flowName} placeholder="My Data Pipeline" />
          </div>

          <div class="form-field">
            <Label for="flowDesc">Description</Label>
            <Input id="flowDesc" bind:value={flowDescription} placeholder="Describe what this flow does..." />
          </div>
        </div>

        <div class="dialog-footer">
          <Button variant="outline" onclick={() => showSaveDialog = false}>Cancel</Button>
          <Button onclick={saveFlow} disabled={isSaving || !flowName}>
            {isSaving ? 'Saving...' : 'Save'}
          </Button>
        </div>
      </div>
    </div>
  {/if}

  <!-- Fullscreen Node Modal -->
  <FullscreenNodeModal {llmProviders} allTools={data.tools} allEvaluations={data.evaluations} onToolUpdated={handleToolUpdated} onAgentUpdated={handleAgentUpdated} onNodeDataUpdated={handleNodeDataUpdated} />


  <!-- Create Tool Modal -->
  {#if showCreateToolModal}
    <div class="create-tool-overlay" onclick={closeCreateToolModal}>
      <div class="create-tool-modal" onclick={(e) => e.stopPropagation()}>
        <div class="create-tool-header">
          <h2 class="create-tool-title">Create New Tool</h2>
          <button class="create-tool-close" onclick={closeCreateToolModal}>×</button>
        </div>

        <div class="create-tool-body">
          <div class="create-tool-section">
            <div class="create-tool-section-label">Tool Details</div>
            <div class="create-tool-form-row">
              <label class="create-tool-label" for="toolName">Name</label>
              <input
                id="toolName"
                class="create-tool-input"
                bind:value={newToolName}
                placeholder="Tool Name"
              />
            </div>
            <div class="create-tool-form-row">
              <label class="create-tool-label" for="toolDesc">Description</label>
              <input
                id="toolDesc"
                class="create-tool-input"
                bind:value={newToolDescription}
                placeholder="Describe what this tool does..."
              />
            </div>
            <div class="create-tool-form-row">
              <label class="create-tool-label">Public</label>
              <div class="create-tool-radio-group">
                <label class="create-tool-radio-label">
                  <input
                    type="radio"
                    name="toolPublic"
                    value={false}
                    checked={!newToolIsPublic}
                    onchange={() => newToolIsPublic = false}
                  />
                  <span>No</span>
                </label>
                <label class="create-tool-radio-label">
                  <input
                    type="radio"
                    name="toolPublic"
                    value={true}
                    checked={newToolIsPublic}
                    onchange={() => newToolIsPublic = true}
                  />
                  <span>Yes</span>
                </label>
              </div>
            </div>

            <div class="create-tool-form-row">
              <button
                class="create-tool-expand-header"
                onclick={() => showWriteWithAI = !showWriteWithAI}
                type="button"
              >
                <span class="expand-icon">{showWriteWithAI ? '∨' : '→'}</span>
                <span>Write with AI</span>
              </button>

              {#if showWriteWithAI}
                <div class="create-tool-ai-expanded">
                  <div class="create-tool-ai-field-left">
                    <label class="create-tool-label" for="aiInstructions">Additional Instructions (optional)</label>
                    <textarea
                      id="aiInstructions"
                      class="create-tool-textarea create-tool-textarea-full"
                      bind:value={additionalInstructions}
                      placeholder="Any additional requirements or constraints..."
                      rows="8"
                    ></textarea>
                  </div>

                  <div class="create-tool-ai-field-right">
                    <div class="create-tool-ai-field">
                      <label class="create-tool-label" for="aiLlmProvider">LLM Provider</label>
                      <select
                        id="aiLlmProvider"
                        class="create-tool-input"
                        bind:value={selectedLLMProvider}
                      >
                        <option value={null}>-- Select LLM --</option>
                        {#each llmProviders as provider}
                          <option value={provider}>{provider.name}</option>
                        {/each}
                      </select>
                      {#if llmProviders.length === 0}
                        <div class="create-tool-helper-text" style="color: #f59e0b;">
                          No LLM providers configured. Configure one in the sidebar's "Attach LLM" panel.
                        </div>
                      {/if}
                    </div>

                    <div class="create-tool-ai-field">
                      <Button
                        onclick={handleGenerateWithAI}
                        disabled={isGeneratingCode || !newToolName.trim() || !newToolDescription.trim() || !selectedLLMProvider}
                        class="bg-purple-600 hover:bg-purple-700"
                      >
                        {#snippet children()}
                          {isGeneratingCode ? 'Writing...' : 'Write with AI'}
                        {/snippet}
                      </Button>
                    </div>
                  </div>
                </div>
              {/if}
            </div>

            <div class="create-tool-form-row">
              <label class="create-tool-label" for="toolMainFunction">Main function name</label>
              <input
                id="toolMainFunction"
                class="create-tool-input"
                bind:value={newToolMainFunction}
                placeholder="e.g. process_data"
              />
              <div class="create-tool-helper-text">
                This should be the name of the top-level function to expose as the tool entrypoint. Its type hints will be used to infer inputs and outputs.
              </div>
            </div>
          </div>

          <div class="create-tool-section">
            <div class="create-tool-section-label">Script Code</div>
            <div class="create-tool-code-container">
              <div bind:this={createToolEditorContainer} class="create-tool-editor-container"></div>
            </div>
          </div>

          <div class="create-tool-section">
            <button
              class="create-tool-expand-header"
              onclick={() => showEditWithAI = !showEditWithAI}
              type="button"
            >
              <span class="expand-icon">{showEditWithAI ? '∨' : '→'}</span>
              <span>Edit with AI</span>
            </button>

            {#if showEditWithAI}
              <div class="create-tool-ai-expanded">
                <div class="create-tool-ai-field-left">
                  <label class="create-tool-label" for="editInstructions">Editing Instructions</label>
                  <textarea
                    id="editInstructions"
                    class="create-tool-textarea create-tool-textarea-full"
                    bind:value={editingInstructions}
                    placeholder="e.g., Add error handling, convert to async/await, refactor to use classes..."
                    rows="8"
                  ></textarea>
                  <div class="create-tool-helper-text">
                    Describe what changes you want to make to the existing code.
                  </div>
                </div>

                <div class="create-tool-ai-field-right">
                  <div class="create-tool-ai-field">
                    <label class="create-tool-label" for="editLlmProvider">LLM Provider</label>
                    <select
                      id="editLlmProvider"
                      class="create-tool-input"
                      bind:value={selectedLLMProvider}
                    >
                      <option value={null}>-- Select LLM --</option>
                      {#each llmProviders as provider}
                        <option value={provider}>{provider.name}</option>
                      {/each}
                    </select>
                    {#if llmProviders.length === 0}
                      <div class="create-tool-helper-text" style="color: #f59e0b;">
                        No LLM providers configured. Configure one in the sidebar's "Attach LLM" panel.
                      </div>
                    {/if}
                  </div>

                  <div class="create-tool-ai-field">
                    <Button
                      onclick={handleEditWithAI}
                      disabled={isEditingCode || !newToolCode.trim() || !editingInstructions.trim() || !selectedLLMProvider}
                      class="bg-green-600 hover:bg-green-700"
                    >
                      {#snippet children()}
                        {isEditingCode ? 'Editing...' : 'Edit with AI'}
                      {/snippet}
                    </Button>
                  </div>
                </div>
              </div>
            {/if}
          </div>
        </div>

        <div class="create-tool-footer">
          <Button variant="outline" onclick={closeCreateToolModal}>
            {#snippet children()}Cancel{/snippet}
          </Button>
          <Button onclick={handleCreateTool} disabled={isCreatingTool} class="bg-blue-600 hover:bg-blue-700">
            {#snippet children()}{isCreatingTool ? 'Creating...' : 'Create Tool'}{/snippet}
          </Button>
        </div>
      </div>
    </div>
  {/if}

  <!-- Create Agent Modal -->
  {#if showCreateAgentModal}
    <div class="create-tool-overlay" onclick={closeCreateAgentModal}>
      <div class="create-tool-modal" onclick={(e) => e.stopPropagation()}>
        <div class="create-tool-header">
          <h2 class="create-tool-title">{isConfiguringComplexNode ? 'Configure Agent Node' : 'Create New Agent'}</h2>
          <button class="create-tool-close" onclick={closeCreateAgentModal}>×</button>
        </div>

        <div class="create-tool-body">
          <!-- Agent Details Section -->
          <div class="create-tool-section">
            <div class="create-tool-section-label">Agent Details</div>
            <div class="create-tool-form-row">
              <label class="create-tool-label" for="agentName">Name</label>
              <input
                id="agentName"
                class="create-tool-input"
                bind:value={newAgentName}
                placeholder="Agent Name"
              />
            </div>
            <div class="create-tool-form-row">
              <label class="create-tool-label" for="agentDesc">Description</label>
              <input
                id="agentDesc"
                class="create-tool-input"
                bind:value={newAgentDescription}
                placeholder="Describe what this agent does..."
              />
            </div>
          </div>

          <!-- System Prompt Section -->
          <div class="create-tool-section">
            <div class="create-tool-section-label">System Prompt</div>
            <textarea
              class="create-tool-textarea"
              bind:value={newAgentSystemPrompt}
              placeholder="Enter the system prompt for this agent..."
              rows="10"
              style="min-height: 200px;"
            ></textarea>
          </div>

          <!-- User Prompt Section -->
          <div class="create-tool-section">
            <div class="create-tool-section-label">User Prompt</div>
            <div class="create-tool-helper-text" style="margin-bottom: 8px;">
              Use &#123;input&#125; where the runtime input should appear. Use &#123;message_history&#125; to include the full conversation history from previous nodes.
            </div>
            <div class="highlighted-textarea-container">
              <div class="highlighted-textarea-backdrop" bind:this={createAgentUserPromptBackdrop} aria-hidden="true">
                {@html newAgentUserPrompt
                  .replace(/&/g, '&amp;')
                  .replace(/</g, '&lt;')
                  .replace(/>/g, '&gt;')
                  .replace(/\{input\}/g, '<span class="template-var">{input}</span>')
                  .replace(/\{message_history\}/g, '<span class="template-var">{message_history}</span>')
                + '\n'}
              </div>
              <textarea
                class="highlighted-textarea"
                bind:value={newAgentUserPrompt}
                placeholder="e.g. &#123;input&#125;"
                rows="4"
                style="min-height: 80px;"
                onscroll={(e) => { if (createAgentUserPromptBackdrop) createAgentUserPromptBackdrop.scrollTop = e.currentTarget.scrollTop; }}
              ></textarea>
            </div>
          </div>

          <!-- Generate with AI Section -->
          <div class="create-tool-section">
            <button
              class="create-tool-expand-header"
              onclick={() => showGeneratePromptAI = !showGeneratePromptAI}
              type="button"
            >
              <span class="expand-icon">{showGeneratePromptAI ? '∨' : '→'}</span>
              <span>Generate with AI</span>
            </button>

            {#if showGeneratePromptAI}
              <div class="create-tool-ai-expanded">
                <div class="create-tool-ai-field-left">
                  <label class="create-tool-label" for="promptInstructions">Additional Instructions (optional)</label>
                  <textarea
                    id="promptInstructions"
                    class="create-tool-textarea create-tool-textarea-full"
                    bind:value={promptAdditionalInstructions}
                    placeholder="Any additional requirements for the system prompt..."
                    rows="8"
                  ></textarea>
                </div>

                <div class="create-tool-ai-field-right">
                  <div class="create-tool-ai-field">
                    <label class="create-tool-label" for="promptLlmProvider">LLM Provider</label>
                    <select
                      id="promptLlmProvider"
                      class="create-tool-input"
                      bind:value={selectedLLMProvider}
                    >
                      <option value={null}>-- Select LLM --</option>
                      {#each llmProviders as provider}
                        <option value={provider}>{provider.name}</option>
                      {/each}
                    </select>
                    {#if llmProviders.length === 0}
                      <div class="create-tool-helper-text" style="color: #f59e0b;">
                        No LLM providers configured. Configure one in the sidebar's "Attach LLM" panel.
                      </div>
                    {/if}
                  </div>

                  <div class="create-tool-ai-field">
                    <Button
                      onclick={handleGeneratePrompt}
                      disabled={isGeneratingPrompt || !newAgentName.trim() || !newAgentDescription.trim() || !selectedLLMProvider}
                      class="bg-purple-600 hover:bg-purple-700"
                    >
                      {#snippet children()}
                        {isGeneratingPrompt ? 'Generating...' : 'Generate with AI'}
                      {/snippet}
                    </Button>
                  </div>
                </div>
              </div>
            {/if}
          </div>

          <!-- LLM Provider Section -->
          <div class="create-tool-section">
            <div class="create-tool-section-label">LLM Provider</div>
            <div class="create-tool-form-row">
              <label class="create-tool-label" for="agentLlmProvider">Select which LLM this agent uses</label>
              <select
                id="agentLlmProvider"
                class="create-tool-input"
                bind:value={newAgentLLMProvider}
              >
                <option value="">-- Select LLM --</option>
                {#each llmProviders as provider}
                  <option value={provider.name}>{provider.name}</option>
                {/each}
              </select>
              <div class="create-tool-helper-text">
                This determines which LLM the agent will use when executing tasks.
              </div>
            </div>
          </div>

          <!-- Select Tools Section -->
          <div class="create-tool-section">
            <div class="create-tool-section-label">Select Tools</div>
            <div class="create-tool-helper-text" style="margin-bottom: 12px;">
              Choose which tools this agent can use.
            </div>
            {#if data.tools.length === 0}
              <div class="create-tool-helper-text" style="color: #f59e0b;">
                No tools available. Create some tools first.
              </div>
            {:else}
              {#each data.tools as tool}
                <label class="tool-checkbox-item">
                  <input
                    type="checkbox"
                    checked={newAgentSelectedTools.includes(tool.id)}
                    onchange={() => toggleToolSelection(tool.id)}
                  />
                  <span class="tool-checkbox-name">{tool.name}</span>
                  {#if tool.description}
                    <span class="tool-checkbox-desc">{tool.description}</span>
                  {/if}
                </label>
              {/each}
            {/if}
          </div>

          <!-- Assign Evaluations Section -->
          <div class="create-tool-section">
            <div class="create-tool-section-label">Assign Evaluations</div>
            <div class="create-tool-helper-text" style="margin-bottom: 12px;">
              Choose which evaluations to run on this agent's output.
            </div>
            {#if data.evaluations.length === 0}
              <div class="create-tool-helper-text" style="color: #94a3b8;">
                No evaluations defined. Create one in the sidebar Evaluations panel.
              </div>
            {:else}
              {#each data.evaluations as evaluation}
                <label class="tool-checkbox-item">
                  <input
                    type="checkbox"
                    checked={newAgentSelectedEvals.includes(evaluation.id)}
                    onchange={() => toggleEvalSelection(evaluation.id)}
                  />
                  <span class="tool-checkbox-name">{evaluation.name}</span>
                  <span class="tool-checkbox-desc">{evaluation.score_type.toLowerCase()}</span>
                </label>
              {/each}
            {/if}
          </div>

          <!-- Output Paths Section -->
          <div class="create-tool-section">
            <div class="create-tool-section-label">Output Paths</div>
            <div class="create-tool-helper-text" style="margin-bottom: 12px;">
              Define conditional output paths for routing in multi-agent workflows.
            </div>
            {#if newAgentOutputPaths.length > 0}
              <div class="output-paths-list">
                {#each newAgentOutputPaths as path, index}
                  <div class="output-path-row">
                    <input
                      type="text"
                      class="create-tool-input output-path-name"
                      bind:value={path.name}
                      placeholder="Path name (e.g., revise)"
                    />
                    <input
                      type="text"
                      class="create-tool-input output-path-description"
                      bind:value={path.description}
                      placeholder="When to choose this path"
                    />
                    <select
                      class="create-tool-input output-path-behavior"
                      bind:value={path.return_behavior}
                    >
                      <option value="node_output">Node Output</option>
                      <option value="previous_output">Previous Output</option>
                    </select>
                    <button class="output-path-remove" onclick={() => removeAgentOutputPath(index)}>
                      &times;
                    </button>
                  </div>
                {/each}
              </div>
            {/if}
            <button class="output-path-add" onclick={addAgentOutputPath} type="button">
              + Add Output Path
            </button>
          </div>
        </div>

        <div class="create-tool-footer">
          <Button variant="outline" onclick={closeCreateAgentModal}>
            {#snippet children()}Cancel{/snippet}
          </Button>
          {#if isConfiguringComplexNode}
            <Button onclick={handleAddNodeToCanvas} class="bg-purple-600 hover:bg-purple-700">
              {#snippet children()}Add to Canvas{/snippet}
            </Button>
          {:else}
            <Button onclick={handleCreateAgent} disabled={isCreatingAgent} class="bg-green-600 hover:bg-green-700">
              {#snippet children()}{isCreatingAgent ? 'Creating...' : 'Create Agent'}{/snippet}
            </Button>
          {/if}
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .app-container {
    display: flex;
    height: 100vh;
    position: relative;
    overflow: hidden;
  }

  .node-window {
    width: 200px;
    background: #f0f0f0;
    padding: 10px;
    border-right: 1px solid #ccc;
    overflow-y: auto;
    overflow-x: hidden;
  }

  .user-section {
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #ccc;
  }

  .current-user {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 8px;
    background: white;
    border-radius: 4px;
    font-size: 12px;
  }

  .current-user strong {
    color: #333;
  }

  .current-user small {
    color: #666;
  }

  .triggers-section {
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #ccc;
  }

  .trigger-draggable {
    border-left: 3px solid #e67e22;
  }

  .flows-section {
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #ccc;
  }

  .flow-item {
    padding: 5px 8px;
    margin: 5px 0;
    background: white;
    border: 1px solid #ccc;
    cursor: pointer;
    transition: background-color 0.2s;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .flow-item:hover {
    background: #e8f4f8;
  }

  .flow-item-name {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .flow-item-delete {
    background: none;
    border: none;
    color: #666;
    font-size: 20px;
    cursor: pointer;
    padding: 0;
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 3px;
    transition: all 0.2s;
    flex-shrink: 0;
  }

  .flow-item-delete:hover {
    background: #ff4444;
    color: white;
  }

  .draggable-node {
    padding: 5px;
    margin: 5px 0;
    background: white;
    border: 1px solid #ccc;
    cursor: grab;
  }

  .sidebar-item {
    padding: 5px 8px;
    margin: 5px 0;
    background: white;
    border: 1px solid #ccc;
    cursor: grab;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .sidebar-item:hover {
    background: #e8f4f8;
  }

  .sidebar-item-name {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .sidebar-item-delete {
    background: none;
    border: none;
    color: #666;
    font-size: 20px;
    cursor: pointer;
    padding: 0;
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 3px;
    transition: all 0.2s;
    flex-shrink: 0;
  }

  .sidebar-item-delete:hover {
    background: #ff4444;
    color: white;
  }

  .flow-container {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    position: relative;
    padding: 20px;
    padding-bottom: 0;
    overflow: hidden;
  }

  .canvas-area {
    flex: 1;
    position: relative;
    min-height: 0;
  }

  .info-toggle-btn {
    position: absolute;
    bottom: 10px;
    left: 10px;
    z-index: 10;
    background: oklch(0.2 0.005 260);
    color: oklch(0.7 0 0);
    border: 1px solid oklch(0.7 0 0 / 0.25);
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 12px;
    font-family: 'SF Mono', 'Fira Code', monospace;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
  }
  .info-toggle-btn:hover {
    background: oklch(0.25 0.005 260);
    color: oklch(0.9 0 0);
  }

  /* Validation Toast */
  .validation-toast {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1000;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 20px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    font-size: 14px;
    font-weight: 500;
    animation: slideIn 0.3s ease-out;
    min-width: 300px;
    max-width: 500px;
  }

  .validation-toast.success {
    background-color: #10b981;
    color: white;
    border: 2px solid #059669;
  }

  .validation-toast.error {
    background-color: #ef4444;
    color: white;
    border: 2px solid #dc2626;
  }

  .toast-icon {
    font-size: 20px;
    font-weight: bold;
  }

  .toast-message {
    flex: 1;
    line-height: 1.4;
  }

  .toast-close {
    background: none;
    border: none;
    color: white;
    font-size: 24px;
    cursor: pointer;
    padding: 0;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 0.8;
    transition: opacity 0.2s;
  }

  .toast-close:hover {
    opacity: 1;
  }

  @keyframes slideIn {
    from {
      transform: translateX(400px);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }

  /* Flow Controls */
  .flow-controls {
    position: absolute;
    top: 20px;
    left: 20px;
    z-index: 10;
    display: flex;
    align-items: center;
  }

  .evals-toggle {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-left: 12px;
    font-size: 12px;
    color: #ccc;
    cursor: pointer;
    user-select: none;
  }
  .evals-toggle input[type="checkbox"] {
    cursor: pointer;
  }
  .evals-toggle span {
    font-weight: 500;
  }

  /* Dialog Overlay */
  .dialog-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10000;
  }

  .dialog-content {
    background: white;
    border-radius: 8px;
    padding: 0;
    min-width: 400px;
    max-width: 500px;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
  }

  .dialog-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    border-bottom: 1px solid #e5e7eb;
  }

  .dialog-header h3 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: #111827;
  }

  .dialog-close {
    background: none;
    border: none;
    font-size: 28px;
    color: #6b7280;
    cursor: pointer;
    padding: 0;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    transition: background-color 0.2s;
  }

  .dialog-close:hover {
    background-color: #f3f4f6;
    color: #111827;
  }

  .dialog-body {
    padding: 24px;
  }

  .form-field {
    margin-bottom: 16px;
  }

  .form-field:last-child {
    margin-bottom: 0;
  }

  .dialog-footer {
    display: flex;
    justify-content: flex-end;
    gap: 12px;
    padding: 16px 24px;
    border-top: 1px solid #e5e7eb;
    background-color: #f9fafb;
    border-bottom-left-radius: 8px;
    border-bottom-right-radius: 8px;
  }

  /* Tools Section */
  .tools-section {
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #ccc;
  }

  /* Create Tool Modal - Dark theme like FullscreenNodeModal */
  .create-tool-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.85);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    animation: fadeIn 0.2s ease-out;
  }

  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  .create-tool-modal {
    background: #1e1e1e;
    border-radius: 8px;
    width: 95vw;
    height: 95vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    animation: slideUp 0.3s ease-out;
  }

  @keyframes slideUp {
    from {
      transform: translateY(50px);
      opacity: 0;
    }
    to {
      transform: translateY(0);
      opacity: 1;
    }
  }

  .create-tool-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    border-bottom: 2px solid #2d2d30;
    background: #252526;
  }

  .create-tool-title {
    margin: 0;
    font-size: 24px;
    font-weight: 600;
    color: #cccccc;
  }

  .create-tool-close {
    background: none;
    border: none;
    font-size: 36px;
    color: #cccccc;
    cursor: pointer;
    padding: 0;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    transition: all 0.2s;
  }

  .create-tool-close:hover {
    background-color: #3e3e42;
    color: #ffffff;
  }

  .create-tool-body {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  .create-tool-section {
    padding: 20px;
    background: #252526;
    border-radius: 8px;
    border: 1px solid #2d2d30;
  }

  .create-tool-section-label {
    font-size: 13px;
    font-weight: 600;
    color: #007acc;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
  }

  .create-tool-form-row {
    margin-bottom: 16px;
  }

  .create-tool-form-row:last-child {
    margin-bottom: 0;
  }

  .create-tool-label {
    display: block;
    font-size: 13px;
    font-weight: 500;
    color: #cccccc;
    margin-bottom: 6px;
  }

  .create-tool-helper-text {
    margin-top: 4px;
    font-size: 12px;
    color: #f5f5f5;
    opacity: 0.9;
  }

  .create-tool-input {
    width: 100%;
    background: #1e1e1e;
    color: #d4d4d4;
    border: 1px solid #2d2d30;
    border-radius: 4px;
    padding: 10px 12px;
    font-size: 14px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    outline: none;
    transition: border-color 0.2s;
  }

  .create-tool-input:focus {
    border-color: #007acc;
  }

  .create-tool-input::placeholder {
    color: #6e6e6e;
  }

  .create-tool-radio-group {
    display: flex;
    gap: 20px;
    align-items: center;
  }

  .create-tool-radio-label {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    font-size: 14px;
    color: #cccccc;
  }

  .create-tool-radio-label input[type="radio"] {
    width: 16px;
    height: 16px;
    cursor: pointer;
    accent-color: #007acc;
  }

  .create-tool-radio-label:hover {
    color: #ffffff;
  }

  .create-tool-expand-header {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #2d2d30;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    padding: 10px 14px;
    width: 100%;
    color: #cccccc;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  }

  .create-tool-expand-header:hover {
    background: #3e3e42;
    color: #ffffff;
  }

  .expand-icon {
    font-size: 12px;
    font-weight: bold;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
  }

  .create-tool-ai-expanded {
    margin-top: 12px;
    padding: 16px;
    background: #1e1e1e;
    border: 1px solid #2d2d30;
    border-radius: 4px;
    display: flex;
    flex-direction: row;
    gap: 16px;
  }

  .create-tool-ai-field-left {
    flex: 1;
    display: flex;
    flex-direction: column;
  }

  .create-tool-ai-field-right {
    width: 280px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .create-tool-ai-field {
    display: flex;
    flex-direction: column;
  }

  .create-tool-textarea {
    width: 100%;
    background: #2d2d30;
    color: #d4d4d4;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    padding: 10px 12px;
    font-size: 14px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    outline: none;
    transition: border-color 0.2s;
    resize: vertical;
  }

  .create-tool-textarea-full {
    flex: 1;
    min-height: 150px;
  }

  .create-tool-textarea:focus {
    border-color: #007acc;
  }

  .create-tool-textarea::placeholder {
    color: #6e6e6e;
  }

  .highlighted-textarea-container {
    position: relative;
    background: #2d2d30;
    border-radius: 4px;
  }

  .highlighted-textarea-backdrop,
  .highlighted-textarea {
    font-family: 'SF Mono', 'Fira Code', 'Fira Mono', Menlo, Consolas, 'DejaVu Sans Mono', monospace;
    font-size: 13px;
    line-height: 1.5;
    letter-spacing: normal;
    word-spacing: normal;
    tab-size: 4;
    padding: 10px 12px;
    white-space: pre-wrap;
    word-wrap: break-word;
    box-sizing: border-box;
    border: 1px solid transparent;
    border-radius: 4px;
    margin: 0;
  }

  .highlighted-textarea-backdrop {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    overflow: auto;
    color: #d4d4d4;
    pointer-events: none;
    scrollbar-width: none;
  }

  .highlighted-textarea-backdrop::-webkit-scrollbar {
    display: none;
  }

  .highlighted-textarea {
    background: transparent !important;
    color: transparent;
    caret-color: #d4d4d4;
    position: relative;
    z-index: 1;
    resize: vertical;
    width: 100%;
    outline: none;
    transition: border-color 0.2s;
    border-color: #3e3e42;
  }

  .highlighted-textarea:focus {
    border-color: #007acc;
  }

  .highlighted-textarea::placeholder {
    color: #6e6e6e;
  }

  .highlighted-textarea::selection {
    background: rgba(0, 122, 204, 0.4);
    color: transparent;
  }

  .highlighted-textarea-backdrop :global(.template-var) {
    color: #4ec9b0;
    font-weight: 600;
  }

  .create-tool-code-container {
    background: #1e1e1e;
    border-radius: 4px;
    overflow: auto;
    border: 1px solid #2d2d30;
    max-height: 400px;
  }

  .create-tool-editor-container {
    min-height: 300px;
    font-size: 14px;
  }

  .create-tool-editor-container :global(.cm-editor) {
    height: 100%;
  }

  .create-tool-editor-container :global(.cm-scroller) {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 14px;
  }

  .create-tool-footer {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 12px;
    padding: 16px 24px;
    border-top: 2px solid #2d2d30;
    background: #252526;
  }

  /* Custom scrollbar for create tool modal */
  .create-tool-body::-webkit-scrollbar {
    width: 12px;
  }

  .create-tool-body::-webkit-scrollbar-track {
    background: #1e1e1e;
  }

  .create-tool-body::-webkit-scrollbar-thumb {
    background: #424242;
    border-radius: 6px;
  }

  .create-tool-body::-webkit-scrollbar-thumb:hover {
    background: #4e4e4e;
  }

  /* Custom scrollbar for node window */
  .node-window::-webkit-scrollbar {
    width: 8px;
  }

  .node-window::-webkit-scrollbar-track {
    background: #e0e0e0;
  }

  .node-window::-webkit-scrollbar-thumb {
    background: #a0a0a0;
    border-radius: 4px;
  }

  .node-window::-webkit-scrollbar-thumb:hover {
    background: #888888;
  }

  /* Agents Section */
  .agents-section {
    margin-bottom: 15px;
    padding-bottom: 10px;
  }

  /* Tool Checkbox Items (for agent tool selection) */
  .tool-checkbox-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 12px;
    margin-bottom: 6px;
    background: #2d2d30;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.2s;
  }

  .tool-checkbox-item:hover {
    background: #3e3e42;
  }

  .tool-checkbox-item input[type="checkbox"] {
    margin-top: 2px;
    width: 16px;
    height: 16px;
    cursor: pointer;
    accent-color: #007acc;
    flex-shrink: 0;
  }

  .tool-checkbox-name {
    font-size: 14px;
    font-weight: 500;
    color: #cccccc;
    white-space: nowrap;
  }

  .tool-checkbox-desc {
    font-size: 12px;
    color: #888888;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* Composed Agent Badge */
  .composed-badge {
    margin-left: 4px;
    font-size: 12px;
  }

  /* Agent Buttons in sidebar */
  .agent-buttons {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  /* Output Paths */
  .output-paths-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .output-path-row {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .output-path-name {
    width: 120px;
  }

  .output-path-description {
    flex: 1;
  }

  .output-path-behavior {
    width: 150px;
    flex-shrink: 0;
  }

  .output-path-remove {
    width: 28px;
    height: 28px;
    border: none;
    background: #5a1d1d;
    color: #ff6b6b;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .output-path-remove:hover {
    background: #7a2d2d;
  }

  .output-path-add {
    padding: 6px 12px;
    background: transparent;
    border: 1px dashed #3e3e42;
    color: #007acc;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    transition: all 0.2s ease;
  }

  .output-path-add:hover {
    border-color: #007acc;
    background: rgba(0, 122, 204, 0.1);
  }
</style>