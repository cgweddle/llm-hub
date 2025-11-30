import { lucia } from "$lib/server/auth";
import { fail, redirect } from "@sveltejs/kit";
import { verify } from "@node-rs/argon2";
import Database from "better-sqlite3";

import type { Actions } from "./$types";

const db = new Database("/Users/chris/Documents/repos/llm-hub/database/llm_hub.db");

export const actions: Actions = {
	default: async (event) => {
		const formData = await event.request.formData();
		const username = formData.get("username");
		const password = formData.get("password");

		if (typeof username !== "string" || username.length < 3 || username.length > 31) {
			return fail(400, {
				message: "Invalid username"
			});
		}
		if (typeof password !== "string" || password.length < 6 || password.length > 255) {
			return fail(400, {
				message: "Invalid password"
			});
		}

		// Find user by username
		const stmt = db.prepare("SELECT id, username, email, password_hash FROM users WHERE username = ?");
		const existingUser = stmt.get(username) as { id: number; username: string; email: string; password_hash: string } | undefined;

		if (!existingUser) {
			return fail(400, {
				message: "Incorrect username or password"
			});
		}

		const validPassword = await verify(existingUser.password_hash, password, {
			memoryCost: 19456,
			timeCost: 2,
			outputLen: 32,
			parallelism: 1
		});

		if (!validPassword) {
			return fail(400, {
				message: "Incorrect username or password"
			});
		}

		const session = await lucia.createSession(existingUser.id, {});
		const sessionCookie = lucia.createSessionCookie(session.id);
		event.cookies.set(sessionCookie.name, sessionCookie.value, {
			path: ".",
			...sessionCookie.attributes
		});

		redirect(302, "/");
	}
};
