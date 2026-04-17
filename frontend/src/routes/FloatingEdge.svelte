<script lang="ts">
  import { getStraightPath, type EdgeProps } from '@xyflow/svelte';

  interface Props {
    source: EdgeProps['source'];
    target: EdgeProps['target'];
    sourceX: EdgeProps['sourceX'];
    sourceY: EdgeProps['sourceY'];
    targetX: EdgeProps['targetX'];
    targetY: EdgeProps['targetY'];
    markerEnd?: EdgeProps['markerEnd'];
    style?: EdgeProps['style'];
    id: EdgeProps['id'];
  }

  let {
    sourceX,
    sourceY,
    targetX,
    targetY,
    markerEnd,
    style,
    id
  }: Props = $props();

  const edgePath = $derived.by(() => {
    try {
      return getStraightPath({ sourceX, sourceY, targetX, targetY })[0];
    } catch (error) {
      console.error('Error calculating edge path:', error);
      return undefined;
    }
  });
</script>

<path {id} marker-end={markerEnd} d={edgePath} {style} />
