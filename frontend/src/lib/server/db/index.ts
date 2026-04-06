import { env } from "$env/dynamic/private";
import type { Adapter } from "lucia";

if (!env.DATABASE_URL) throw new Error("DATABASE_URL is not set");

const isPostgres = env.DATABASE_URL.startsWith("postgres");

// These are typed as `any` because the concrete type depends on the dialect.
// Query results are still fully typed via Drizzle's inference.
let db: any;
let adapter: Adapter;
let users: any;
let sessions: any;

if (isPostgres) {
	const { drizzle } = await import("drizzle-orm/postgres-js");
	const postgres = (await import("postgres")).default;
	const { DrizzlePostgreSQLAdapter } = await import("@lucia-auth/adapter-drizzle");
	const schema = await import("./schema.pg.js");

	const client = postgres(env.DATABASE_URL);
	db = drizzle(client, { schema });
	users = schema.users;
	sessions = schema.sessions;
	adapter = new DrizzlePostgreSQLAdapter(db, sessions, users);
} else {
	const { drizzle } = await import("drizzle-orm/better-sqlite3");
	const Database = (await import("better-sqlite3")).default;
	const { DrizzleSQLiteAdapter } = await import("@lucia-auth/adapter-drizzle");
	const schema = await import("./schema.sqlite.js");

	const client = new Database(env.DATABASE_URL);
	client.pragma("journal_mode = WAL");
	db = drizzle(client, { schema });
	users = schema.users;
	sessions = schema.sessions;
	adapter = new DrizzleSQLiteAdapter(db, sessions, users);
}

export { db, adapter, users, sessions };
