import { writable } from 'svelte/store';

export type BuilderMode = 'flow' | 'agent';

function createBuilderModeStore() {
  const { subscribe, set, update } = writable<BuilderMode>('flow');

  return {
    subscribe,
    setFlow: () => set('flow'),
    setAgent: () => set('agent'),
    toggle: () => update(mode => mode === 'flow' ? 'agent' : 'flow'),
  };
}

export const builderMode = createBuilderModeStore();
