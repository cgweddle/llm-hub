<script lang="ts">
	import { Select as SelectPrimitive } from "bits-ui";
	import { cn } from "$lib/utils.js";

	let {
		ref = $bindable(null),
		class: className,
		children,
		...restProps
	}: SelectPrimitive.RootProps & {
		ref?: HTMLButtonElement | null;
		class?: string;
		children?: import('svelte').Snippet;
	} = $props();
</script>

<SelectPrimitive.Root {...restProps}>
	{@render children?.()}
</SelectPrimitive.Root>
