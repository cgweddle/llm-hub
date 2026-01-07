<script lang="ts">
	import { Select as SelectPrimitive } from "bits-ui";
	import { cn } from "$lib/utils.js";

	let {
		ref = $bindable(null),
		class: className,
		children,
		...restProps
	}: SelectPrimitive.TriggerProps & {
		ref?: HTMLButtonElement | null;
		class?: string;
		children?: import('svelte').Snippet;
	} = $props();
</script>

<SelectPrimitive.Trigger
	bind:ref
	class={cn(
		"flex h-9 w-full items-center justify-between whitespace-nowrap rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-50 [&>span]:line-clamp-1",
		className
	)}
	{...restProps}
>
	{@render children?.()}
</SelectPrimitive.Trigger>
