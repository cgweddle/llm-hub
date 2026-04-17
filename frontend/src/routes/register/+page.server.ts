import { lucia } from "$lib/server/auth";
import { fail, redirect } from "@sveltejs/kit";
import { hash } from "@node-rs/argon2";
import { db, users } from "$lib/server/db";
import { eq } from "drizzle-orm";

import type { Actions } from "./$types";

export const actions: Actions = {
	default: async (event) => {
		const formData = await event.request.formData();
		const username = formData.get("username");
		const email = formData.get("email");
		const password = formData.get("password");

		// Validation
		if (
			typeof username !== "string" ||
			username.length < 3 ||
			username.length > 31 ||
			!/^[a-z0-9_-]+$/.test(username)
		) {
			return fail(400, {
				message: "Invalid username. Use 3-31 characters, lowercase letters, numbers, _ or -"
			});
		}
		if (typeof email !== "string" || email.length < 3 || email.length > 255) {
			return fail(400, {
				message: "Invalid email"
			});
		}
		if (typeof password !== "string" || password.length < 6 || password.length > 255) {
			return fail(400, {
				message: "Invalid password. Must be at least 6 characters"
			});
		}

		const passwordHash = await hash(password, {
			memoryCost: 19456,
			timeCost: 2,
			outputLen: 32,
			parallelism: 1
		});

		try {
			// Insert user and retrieve the ID
			await db.insert(users).values({
				username,
				email,
				passwordHash
			});

			// Query back to get the auto-generated ID
			const [newUser] = await db
				.select({ id: users.id })
				.from(users)
				.where(eq(users.username, username));

			// Create session with Lucia
			const session = await lucia.createSession(newUser.id, {});
			const sessionCookie = lucia.createSessionCookie(session.id);
			event.cookies.set(sessionCookie.name, sessionCookie.value, {
				path: ".",
				...sessionCookie.attributes
			});
		} catch (e: any) {
			console.error("Registration failed:", e);
			// Detect unique constraint violations across Postgres (code '23505') and SQLite (message text)
			const isUniqueViolation =
				e.code === "23505" ||
				(e.message?.toLowerCase() ?? "").match(/unique|duplicate|constraint/);
			if (isUniqueViolation) {
				const fieldHint = (e.detail ?? e.constraint_name ?? e.message ?? "").toLowerCase();
				if (fieldHint.includes("username")) {
					return fail(400, { message: "Username already taken" });
				}
				if (fieldHint.includes("email")) {
					return fail(400, { message: "Email already registered" });
				}
				return fail(400, { message: "That username or email is already taken" });
			}
			return fail(500, {
				message: "An error occurred during registration"
			});
		}

		redirect(302, "/");
	}
};
