<script lang="ts">
	import { Select as SelectPrimitive } from "bits-ui";
	import { cn } from "$lib/utils.js";

	let {
		ref = $bindable(null),
		class: className,
		value,
		label,
		disabled = false,
		children,
		...restProps
	}: SelectPrimitive.ItemProps & {
		ref?: HTMLDivElement | null;
		class?: string;
		children?: import('svelte').Snippet;
	} = $props();
</script>

<SelectPrimitive.Item
	bind:ref
	{value}
	{disabled}
	{label}
	class={cn(
		"relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-2 pr-8 text-sm outline-none data-[disabled]:pointer-events-none data-[highlighted]:bg-accent data-[highlighted]:text-accent-foreground data-[disabled]:opacity-50",
		className
	)}
	{...restProps}
>
	{@render children?.()}
</SelectPrimitive.Item>
