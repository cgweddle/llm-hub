<script lang="ts">
	import { Button } from "$lib/components/ui/button";
	import { Input } from "$lib/components/ui/input";
	import { Label } from "$lib/components/ui/label";
	import { enhance } from "$app/forms";
	import type { ActionData } from "./$types";

	export let form: ActionData;
</script>

<div class="flex min-h-screen items-center justify-center bg-background">
	<div class="w-full max-w-md space-y-6 rounded-lg border bg-card p-8 shadow-lg">
		<div class="space-y-2 text-center">
			<h1 class="text-3xl font-bold">Create Account</h1>
			<p class="text-muted-foreground">Enter your details to get started</p>
		</div>

		{#if form?.message}
			<div class="bg-destructive/15 text-destructive px-4 py-3 rounded-md text-sm">
				{form.message}
			</div>
		{/if}

		<form method="POST" use:enhance class="space-y-4">
			<div class="space-y-2">
				<Label for="username">Username</Label>
				<Input
					id="username"
					name="username"
					type="text"
					required
					placeholder="username"
					autocomplete="username"
				/>
			</div>

			<div class="space-y-2">
				<Label for="email">Email</Label>
				<Input
					id="email"
					name="email"
					type="email"
					required
					placeholder="you@example.com"
					autocomplete="email"
				/>
			</div>

			<div class="space-y-2">
				<Label for="password">Password</Label>
				<Input
					id="password"
					name="password"
					type="password"
					required
					placeholder="••••••••"
					autocomplete="new-password"
				/>
			</div>

			<Button type="submit" class="w-full">Create Account</Button>
		</form>

		<div class="text-center text-sm">
			Already have an account?
			<a href="/login" class="text-primary hover:underline">Sign in</a>
		</div>
	</div>
</div>
