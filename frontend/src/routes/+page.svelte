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
  import { onDestroy, setContext } from 'svelte';

  import ColorSelectorNode from './ColorSelectorNode.svelte';
  import ToolNode from './ToolNode.svelte';
  import FloatingEdge from './FloatingEdge.svelte';
  import CondaEnvironmentsPanel from './CondaEnvironmentsPanel.svelte';
  import LLMProvidersPanel from './LLMProvidersPanel.svelte';
  import FullscreenNodeModal from './FullscreenNodeModal.svelte';
  import { Button } from "$lib/components/ui/button";
  import { Input } from "$lib/components/ui/input";
  import { Label } from "$lib/components/ui/label";
  import {
    validateTwoTools,
    createFlow,
    executeFlow,
    getFlowDetails,
    createPythonScriptTool,
    generateToolCodeStream,
    editToolCodeStream,
    type ValidationResult,
    type Tool,
    type Agent,
    type FlowCreateRequest,
    type Flow as FlowType,
    type CodeGenerateRequest
  } from '../lib/api';
  import { buildEnhancedGraphConfig } from '$lib/flowBuilder';
  import { autoLayoutNodes } from '$lib/elkLayout';
  import '@xyflow/svelte/dist/style.css';
  import type { PageData } from './$types';

  export let data: PageData;

  // Track viewport for coordinate conversion
  let viewport: Viewport = { x: 0, y: 0, zoom: 1 };

  // Validation state
  let validationMessage = '';
  let showValidationToast = false;
  let validationSuccess = false;

  // Flow save state
  let flowName = '';
  let flowDescription = '';
  let showSaveDialog = false;
  let isSaving = false;

  // Conda environment state
  let selectedCondaEnv: string | null = null;

  // LLM provider state
  import type { LLMProvider } from '$lib/store';
  let selectedLLMProvider: LLMProvider | null = null;
  let llmProviders: LLMProvider[] = [];

  // Make llmProviders available to child components (ToolNode) as a writable store
  const llmProvidersStore = writable<LLMProvider[]>([]);
  setContext('llmProviders', llmProvidersStore);

  // Sync llmProviders array with the store
  $: llmProvidersStore.set(llmProviders);

  // Create Tool modal state
  let showCreateToolModal = false;
  let newToolName = '';
  let newToolDescription = '';
  let newToolCode = '';
  let newToolMainFunction = '';
  let newToolIsPublic = false;
  let showWriteWithAI = false;
  let additionalInstructions = '';
  let showEditWithAI = false;
  let editingInstructions = '';
  let isCreatingTool = false;
  let isGeneratingCode = false;
  let isEditingCode = false;

  // CodeMirror editor for Create Tool modal
  let createToolEditorContainer: HTMLDivElement;
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
  $: if (showCreateToolModal && createToolEditorContainer) {
    initCreateToolEditor();
  }

  // Cleanup on component destroy
  onDestroy(() => {
    destroyCreateToolEditor();
  });

  const nodeTypes = {
    selectorNode: ColorSelectorNode,
    toolNode: ToolNode
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

  // Simple working example
  let nodes: Node[] = [
    {
      id: '1',
      type: 'input',
      data: { label: 'Input' },
      position: { x: 100, y: 100 },
      sourcePosition: Position.Right
    },
    {
      id: '2',
      type: 'selectorNode',
      data: { color: writable('#ff0000'), handles: ['a', 'b'] },
      position: { x: 400, y: 100 },
      sourcePosition: Position.Right,
      targetPosition: Position.Left
    },
    {
      id: '3',
      type: 'output',
      data: { label: 'Output' },
      position: { x: 700, y: 100 },
      targetPosition: Position.Left
    }
  ];

  let edges: Edge[] = [];

  // Use tools and agents from database instead of hardcoded nodes
  $: availableTools = data.tools.map(tool => tool.name);
  $: availableAgents = data.agents.map(agent => agent.name);
  $: availableFlows = data.flows;

  function addNode(nodeName: string, position: { x: number; y: number }) {
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

  async function onConnect(params) {
    try {
      console.log('Connecting:', params);

      // Get source and target nodes
      const sourceNode = nodes.find(n => n.id === params.source);
      const targetNode = nodes.find(n => n.id === params.target);

      // If both nodes have tool IDs, validate compatibility
      if (sourceNode?.data?.toolId && targetNode?.data?.toolId) {
        const validation = await validateTwoTools(
          sourceNode.data.toolId,
          targetNode.data.toolId
        );

        if (!validation.compatible) {
          // Show error toast
          validationSuccess = false;
          validationMessage = `Incompatible tools: ${validation.issues.join(', ')}`;
          showValidationToast = true;
          setTimeout(() => { showValidationToast = false; }, 5000);
          return; // Don't create the connection
        } else {
          // Show success toast
          validationSuccess = true;
          validationMessage = 'Tools are compatible!';
          showValidationToast = true;
          setTimeout(() => { showValidationToast = false; }, 3000);
        }
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

      // Create flow request
      const flowData: FlowCreateRequest = {
        name: flowName,
        description: flowDescription,
        graph_config: graphConfig,
        is_public: false,
        user_id: data.user?.id || 1,  // Use user ID or default to 1
        conda_env: selectedCondaEnv || undefined  // Store conda env as separate field
      };

      // Send to backend
      const createdFlow = await createFlow(flowData);

      // Show success
      validationSuccess = true;
      validationMessage = `Flow "${createdFlow.name}" saved successfully!`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 3000);

      // Reset and close dialog
      showSaveDialog = false;
      flowName = '';
      flowDescription = '';

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
                input_schema: tool.input_schema,
                output_schema: tool.output_schema,
                runtimeLLM: runtimeLLM
              },
              position: { x: 100 + Math.random() * 400, y: 100 + Math.random() * 300 },
              sourcePosition: Position.Right,
              targetPosition: Position.Left
            };
            nodeMap.set(nodeId, newNode);
          }
        }
        // TODO: Add agent node support
      }

      // Recreate edges from graph_config with handle information
      const newEdges: Edge[] = [];
      for (const edgeConfig of graphConfig.edges) {
        if (nodeMap.has(edgeConfig.from_node) && nodeMap.has(edgeConfig.to_node)) {
          const edge: Edge = {
            id: `${edgeConfig.from_node}-${edgeConfig.to_node}`,
            source: edgeConfig.from_node,
            target: edgeConfig.to_node,
            ...defaultEdgeOptions
          };

          // Add sourceHandle and targetHandle if mapping exists
          if (edgeConfig.mapping && Object.keys(edgeConfig.mapping).length > 0) {
            // For now, use the first mapping entry
            // TODO: Handle multiple output→input mappings (might need multiple edges)
            const [outputField, inputParam] = Object.entries(edgeConfig.mapping)[0];
            edge.sourceHandle = `output-${outputField}`;
            edge.targetHandle = `input-${inputParam}`;
          }

          newEdges.push(edge);
        }
      }

      // Auto-layout the nodes using ELK
      const layoutedNodes = await autoLayoutNodes(Array.from(nodeMap.values()), newEdges);
      nodes = layoutedNodes;
      edges = newEdges;

      // Set the conda environment from the loaded flow
      selectedCondaEnv = flow.conda_env;

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
   * Execute a saved flow
   */
  async function runFlow(flowId: number, initialInput: Record<string, any>) {
    try {
      const result = await executeFlow(flowId, initialInput, selectedCondaEnv);

      if (result.status === 'completed') {
        console.log('✓ Flow completed successfully');
        console.log('Final output:', result.final_output);
        console.log('Execution trace:', result.execution_trace);

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
          provider: selectedLLMProvider.provider,
          model: selectedLLMProvider.model,
          api_key: selectedLLMProvider.apiKey,
          base_url: selectedLLMProvider.baseUrl,
          additional_instructions: additionalInstructions.trim() || undefined
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
          provider: selectedLLMProvider.provider,
          model: selectedLLMProvider.model,
          api_key: selectedLLMProvider.apiKey,
          base_url: selectedLLMProvider.baseUrl
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
</script>

<div class="app-container">
  <!-- Validation Toast -->
  {#if showValidationToast}
    <div class="validation-toast" class:success={validationSuccess} class:error={!validationSuccess}>
      <span class="toast-icon">{validationSuccess ? '✓' : '✗'}</span>
      <span class="toast-message">{validationMessage}</span>
      <button class="toast-close" on:click={() => showValidationToast = false}>×</button>
    </div>
  {/if}

  <div class="node-window">
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

    <LLMProvidersPanel bind:selectedProvider={selectedLLMProvider} bind:providers={llmProviders} />

    <div class="flows-section">
      <h4>Available Flows</h4>
      <Button class="w-full mb-2" size="sm" on:click={() => showSaveDialog = true}>
        Create New Flow
      </Button>
      {#each availableFlows as flow}
        <div
          class="flow-item"
          on:click={() => loadFlow(flow.id)}
        >
          {flow.name}
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
      {#each availableTools as tool}
        <div
          class="draggable-node"
          draggable="true"
          on:dragstart={(event) => event.dataTransfer?.setData('text/plain', tool)}
        >
          {tool}
        </div>
      {/each}
    </div>

    <h4>Available Agents</h4>
    {#each availableAgents as agent}
      <div
        class="draggable-node"
        draggable="true"
        on:dragstart={(event) => event.dataTransfer?.setData('text/plain', agent)}
      >
        {agent}
      </div>
    {/each}
  </div>

  <div
    class="flow-container"
    role="application"
    on:dragover={(event) => event.preventDefault()}
    on:drop={(event) => {
      event.preventDefault();
      const nodeName = event.dataTransfer?.getData('text/plain');
      if (!nodeName) return;

      // Get the flow container bounds
      const flowContainer = event.currentTarget;
      const rect = flowContainer.getBoundingClientRect();

      // Convert screen coordinates to flow coordinates using viewport
      const position = {
        x: (event.clientX - rect.left - viewport.x) / viewport.zoom,
        y: (event.clientY - rect.top - viewport.y) / viewport.zoom
      };

      addNode(nodeName, position);
    }}
  >
    <!-- Flow Controls -->
    <div class="flow-controls">
      <Button on:click={() => showSaveDialog = true}>
        Save Flow
      </Button>
    </div>

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
      on:connect={onConnect}
    >
      <Background />
      <Controls />
      <MiniMap />
    </SvelteFlow>
  </div>

  <!-- Save Flow Dialog -->
  {#if showSaveDialog}
    <div class="dialog-overlay" on:click={() => showSaveDialog = false}>
      <div class="dialog-content" on:click={(e) => e.stopPropagation()}>
        <div class="dialog-header">
          <h3>Save Flow</h3>
          <button class="dialog-close" on:click={() => showSaveDialog = false}>×</button>
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
          <Button variant="outline" on:click={() => showSaveDialog = false}>Cancel</Button>
          <Button on:click={saveFlow} disabled={isSaving || !flowName}>
            {isSaving ? 'Saving...' : 'Save'}
          </Button>
        </div>
      </div>
    </div>
  {/if}

  <!-- Fullscreen Node Modal -->
  <FullscreenNodeModal {llmProviders} />

  <!-- Create Tool Modal -->
  {#if showCreateToolModal}
    <div class="create-tool-overlay" on:click={closeCreateToolModal}>
      <div class="create-tool-modal" on:click={(e) => e.stopPropagation()}>
        <div class="create-tool-header">
          <h2 class="create-tool-title">Create New Tool</h2>
          <button class="create-tool-close" on:click={closeCreateToolModal}>×</button>
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
                    on:change={() => newToolIsPublic = false}
                  />
                  <span>No</span>
                </label>
                <label class="create-tool-radio-label">
                  <input
                    type="radio"
                    name="toolPublic"
                    value={true}
                    checked={newToolIsPublic}
                    on:change={() => newToolIsPublic = true}
                  />
                  <span>Yes</span>
                </label>
              </div>
            </div>

            <div class="create-tool-form-row">
              <button
                class="create-tool-expand-header"
                on:click={() => showWriteWithAI = !showWriteWithAI}
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
              on:click={() => showEditWithAI = !showEditWithAI}
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
</div>

<style>
  .app-container {
    display: flex;
    height: 100vh;
    position: relative;
  }

  .node-window {
    width: 200px;
    background: #f0f0f0;
    padding: 10px;
    border-right: 1px solid #ccc;
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

  .flows-section {
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #ccc;
  }

  .flow-item {
    padding: 5px;
    margin: 5px 0;
    background: white;
    border: 1px solid #ccc;
    cursor: pointer;
    transition: background-color 0.2s;
  }

  .flow-item:hover {
    background: #e8f4f8;
  }

  .draggable-node {
    padding: 5px;
    margin: 5px 0;
    background: white;
    border: 1px solid #ccc;
    cursor: grab;
  }

  .flow-container {
    flex-grow: 1;
    position: relative;
    padding: 20px;
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
    z-index: 1001;
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
</style>