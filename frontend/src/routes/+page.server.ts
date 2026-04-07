import type { PageServerLoad } from "./$types";
import { env } from '$env/dynamic/private';

export const load: PageServerLoad = async (event) => {
	const API_BASE = env.BACKEND_URL || 'http://localhost:8000';

	try {
		let tools, flows, agents, evaluations;

		if (event.locals.user) {
			// User is logged in - fetch their tools/agents + public tools/agents
			const userId = event.locals.user.id;
			const [toolsResponse, flowsResponse, agentsResponse, evalsResponse] = await Promise.all([
				fetch(`${API_BASE}/tools/available/${userId}`),
				fetch(`${API_BASE}/flows/available/${userId}`),
				fetch(`${API_BASE}/agents/available/${userId}`),
				fetch(`${API_BASE}/evaluations/?user_id=${userId}`)
			]);

			tools = toolsResponse.ok ? await toolsResponse.json() : [];
			flows = flowsResponse.ok ? await flowsResponse.json() : [];
			agents = agentsResponse.ok ? await agentsResponse.json() : [];
			evaluations = evalsResponse.ok ? await evalsResponse.json() : [];
		} else {
			// User is not logged in - fetch only public tools/agents
			const [toolsResponse, flowsResponse, agentsResponse] = await Promise.all([
				fetch(`${API_BASE}/tools/public`),
				fetch(`${API_BASE}/flows/public`),
				fetch(`${API_BASE}/agents/public`)
			]);

			tools = toolsResponse.ok ? await toolsResponse.json() : [];
			flows = flowsResponse.ok ? await flowsResponse.json() : [];
			agents = agentsResponse.ok ? await agentsResponse.json() : [];
			evaluations = [];
		}

		return {
			user: event.locals.user,
			tools,
			flows,
			agents,
			evaluations
		};
	} catch (error) {
		console.error('Error fetching tools/flows/agents:', error);
		return {
			user: event.locals.user,
			tools: [],
			flows: [],
			agents: [],
			evaluations: []
		};
	}
};
