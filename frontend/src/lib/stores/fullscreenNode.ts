import { writable } from 'svelte/store';

export interface FullscreenNodeData {
  nodeId: string;
  nodeType: 'tool' | 'expandable' | 'colorSelector';
  data: any;
}

function createFullscreenNodeStore() {
  const { subscribe, set, update } = writable<FullscreenNodeData | null>(null);

  return {
    subscribe,
    open: (nodeData: FullscreenNodeData) => set(nodeData),
    close: () => set(null),
  };
}

export const fullscreenNode = createFullscreenNodeStore();
