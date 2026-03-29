<script lang="ts">
	import { Select as SelectPrimitive } from "bits-ui";
	import { cn } from "$lib/utils.js";
	import { flyAndScale } from "$lib/utils.js";

	let {
		ref = $bindable(null),
		class: className,
		children,
		...restProps
	}: SelectPrimitive.ContentProps & {
		ref?: HTMLDivElement | null;
		class?: string;
		children?: import('svelte').Snippet;
	} = $props();
</script>

<SelectPrimitive.Content
	bind:ref
	transition={flyAndScale}
	class={cn(
		"relative z-50 max-h-96 min-w-[8rem] overflow-hidden rounded-md border bg-popover text-popover-foreground shadow-md outline-none",
		className
	)}
	{...restProps}
>
	<div class="w-full p-1">
		{@render children?.()}
	</div>
</SelectPrimitive.Content>
