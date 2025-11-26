<script lang="ts">
  import { createUser, type UserCreate } from '../lib/api';
  import * as Dialog from "$lib/components/ui/dialog";
  import { Button } from "$lib/components/ui/button";
  import { Input } from "$lib/components/ui/input";
  import { Label } from "$lib/components/ui/label";

  export let open: boolean;
  export let onUserCreated: (user: any) => void;

  let username = '';
  let email = '';
  let password = '';
  let confirmPassword = '';
  let errorMessage = '';
  let isSubmitting = false;

  function resetForm() {
    username = '';
    email = '';
    password = '';
    confirmPassword = '';
    errorMessage = '';
  }

  async function handleSubmit() {
    errorMessage = '';

    if (!username || !email || !password) {
      errorMessage = 'All fields are required';
      return;
    }

    if (password !== confirmPassword) {
      errorMessage = 'Passwords do not match';
      return;
    }

    if (password.length < 6) {
      errorMessage = 'Password must be at least 6 characters';
      return;
    }

    isSubmitting = true;

    try {
      const user = await createUser({ username, email, password });
      onUserCreated(user);
      resetForm();
      open = false;
    } catch (error) {
      errorMessage = error instanceof Error ? error.message : 'Failed to create user';
    } finally {
      isSubmitting = false;
    }
  }
</script>

<Dialog.Root bind:open>
  <Dialog.Content class="sm:max-w-[425px]">
    <Dialog.Header>
      <Dialog.Title>Create New User</Dialog.Title>
      <Dialog.Description>
        Enter your details to create a new user account.
      </Dialog.Description>
    </Dialog.Header>

    {#if errorMessage}
      <div class="bg-destructive/15 text-destructive px-4 py-3 rounded-md text-sm">
        {errorMessage}
      </div>
    {/if}

    <form on:submit|preventDefault={handleSubmit} class="space-y-4">
      <div class="space-y-2">
        <Label for="username">Username</Label>
        <Input
          id="username"
          type="text"
          bind:value={username}
          required
          disabled={isSubmitting}
          placeholder="Enter username"
        />
      </div>

      <div class="space-y-2">
        <Label for="email">Email</Label>
        <Input
          id="email"
          type="email"
          bind:value={email}
          required
          disabled={isSubmitting}
          placeholder="Enter email"
        />
      </div>

      <div class="space-y-2">
        <Label for="password">Password</Label>
        <Input
          id="password"
          type="password"
          bind:value={password}
          required
          disabled={isSubmitting}
          placeholder="Enter password"
        />
      </div>

      <div class="space-y-2">
        <Label for="confirmPassword">Confirm Password</Label>
        <Input
          id="confirmPassword"
          type="password"
          bind:value={confirmPassword}
          required
          disabled={isSubmitting}
          placeholder="Confirm password"
        />
      </div>

      <Dialog.Footer>
        <Button type="button" variant="outline" onclick={() => open = false} disabled={isSubmitting}>
          Cancel
        </Button>
        <Button type="submit" disabled={isSubmitting}>
          {isSubmitting ? 'Creating...' : 'Create User'}
        </Button>
      </Dialog.Footer>
    </form>
  </Dialog.Content>
</Dialog.Root>
