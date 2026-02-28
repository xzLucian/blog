<template>
  <node-view-wrapper class="callout-block" :class="`callout-block--${node.attrs.type}`" data-drag-handle>
    <div class="callout-block__header" contenteditable="false">
      <span class="callout-block__label">{{ node.attrs.title || node.attrs.type.toUpperCase() }}</span>
    </div>
    <node-view-content class="callout-block__body" />
  </node-view-wrapper>
</template>

<script setup lang="ts">
import { NodeViewWrapper, NodeViewContent, nodeViewProps } from '@tiptap/vue-3'

const props = defineProps(nodeViewProps)
</script>
