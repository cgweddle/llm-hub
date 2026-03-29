<script lang="ts">
	import { Select as SelectPrimitive } from "bits-ui";
	import { cn } from "$lib/utils.js";

	let {
		class: className,
		placeholder = "Select an option",
		...restProps
	}: SelectPrimitive.ValueProps & {
		class?: string;
	} = $props();
</script>

<SelectPrimitive.Value
	class={cn("text-sm", className)}
	{placeholder}
	{...restProps}
/>
