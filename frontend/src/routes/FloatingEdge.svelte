<svelte:options immutable />

<script lang="ts">
  import { getStraightPath, type EdgeProps } from '@xyflow/svelte';

  type $$Props = EdgeProps;

  export let source: EdgeProps['source'];
  export let target: EdgeProps['target'];
  export let sourceX: EdgeProps['sourceX'];
  export let sourceY: EdgeProps['sourceY'];
  export let targetX: EdgeProps['targetX'];
  export let targetY: EdgeProps['targetY'];
  export let markerEnd: EdgeProps['markerEnd'] = undefined;
  export let style: EdgeProps['style'] = undefined;
  export let id: EdgeProps['id'];

  let edgePath: string | undefined;

  $: {
    try {
      const pathResult = getStraightPath({
        sourceX,
        sourceY,
        targetX,
        targetY
      });
      edgePath = pathResult[0];
    } catch (error) {
      console.error('Error calculating edge path:', error);
      edgePath = undefined;
    }
  }
</script>

<path {id} marker-end={markerEnd} d={edgePath} {style} />
