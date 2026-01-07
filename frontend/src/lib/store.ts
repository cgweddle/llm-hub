import { writable } from 'svelte/store';
import type { Agent, Tool, Flow } from './api.ts';

// LLM Provider types
export type LLMProviderType = 'anthropic' | 'openai' | 'gemini' | 'lmstudio';

export interface LLMProvider {
  name: string;
  provider: LLMProviderType;
  apiKey?: string;
  baseUrl?: string;
  model: string;
}

// Current user (you might want to implement proper authentication)
export const currentUser = writable<{ id: number; username: string } | null>(null);

// LLM Providers store
export const llmProviders = writable<LLMProvider[]>([]);
export const selectedLLMProvider = writable<LLMProvider | null>(null);

// Available items stores
export const availableAgents = writable<Agent[]>([]);
export const availableTools = writable<Tool[]>([]);
export const availableFlows = writable<Flow[]>([]);

// Loading states
export const isLoadingAgents = writable(false);
export const isLoadingTools = writable(false);
export const isLoadingFlows = writable(false);

// Error states
export const error = writable<string | null>(null);

// Function to clear all data
export function clearAllData() {
  availableAgents.set([]);
  availableTools.set([]);
  availableFlows.set([]);
  error.set(null);
}

// Function to set error
export function setError(message: string | null) {
  error.set(message);
}
