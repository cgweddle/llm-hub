import { Lucia } from "lucia";
import { BetterSqlite3Adapter } from "@lucia-auth/adapter-sqlite";
import Database from "better-sqlite3";
import { dev } from "$app/environment";

// Path to your existing database
const db = new Database("/Users/chris/Documents/repos/llm-hub/database/llm_hub.db");

// Create adapter using your existing users table and new sessions table
const adapter = new BetterSqlite3Adapter(db, {
	user: "users",
	session: "sessions"
});

export const lucia = new Lucia(adapter, {
	sessionCookie: {
		attributes: {
			// set to `true` when using HTTPS
			secure: !dev
		}
	},
	getUserAttributes: (attributes) => {
		return {
			username: attributes.username,
			email: attributes.email
		};
	}
});

declare module "lucia" {
	interface Register {
		Lucia: typeof lucia;
		DatabaseUserAttributes: DatabaseUserAttributes;
	}
}

interface DatabaseUserAttributes {
	username: string;
	email: string;
}
