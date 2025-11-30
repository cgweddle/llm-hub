import type { PageServerLoad } from "./$types";

export const load: PageServerLoad = async (event) => {
	const API_BASE = 'http://localhost:8000';

	try {
		let tools, flows;

		if (event.locals.user) {
			// User is logged in - fetch their tools + public tools
			const userId = event.locals.user.id;
			const [toolsResponse, flowsResponse] = await Promise.all([
				fetch(`${API_BASE}/tools/available/${userId}`),
				fetch(`${API_BASE}/flows/available/${userId}`)
			]);

			tools = toolsResponse.ok ? await toolsResponse.json() : [];
			flows = flowsResponse.ok ? await flowsResponse.json() : [];
		} else {
			// User is not logged in - fetch only public tools
			const [toolsResponse, flowsResponse] = await Promise.all([
				fetch(`${API_BASE}/tools/public`),
				fetch(`${API_BASE}/flows/public`)
			]);

			tools = toolsResponse.ok ? await toolsResponse.json() : [];
			flows = flowsResponse.ok ? await flowsResponse.json() : [];
		}

		return {
			user: event.locals.user,
			tools,
			flows
		};
	} catch (error) {
		console.error('Error fetching tools/flows:', error);
		return {
			user: event.locals.user,
			tools: [],
			flows: []
		};
	}
};
