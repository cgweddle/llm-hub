import { sqliteTable, text, integer } from "drizzle-orm/sqlite-core";

export const users = sqliteTable("users", {
	id: integer("id").primaryKey({ autoIncrement: true }),
	username: text("username", { length: 50 }).notNull().unique(),
	email: text("email", { length: 120 }).notNull().unique(),
	passwordHash: text("password_hash", { length: 255 }).notNull(),
	createdAt: text("created_at"),
	updatedAt: text("updated_at"),
	isActive: integer("is_active", { mode: "boolean" }).default(true)
});

export const sessions = sqliteTable("sessions", {
	id: text("id").primaryKey(),
	userId: integer("user_id")
		.notNull()
		.references(() => users.id),
	expiresAt: integer("expires_at").notNull()
});
