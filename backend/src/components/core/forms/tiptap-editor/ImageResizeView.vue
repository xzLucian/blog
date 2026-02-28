<template>
  <NodeViewWrapper class="image-view" :class="`image-align-${alignment}`" :data-align="alignment">
    <div
      class="image-container"
      :class="{ 'image-container--selected': props.selected }"
      @mouseenter="hovered = true"
      @mouseleave="hovered = false"
    >
      <div v-show="hovered || props.selected" class="image-toolbar" @mousedown.stop>
        <button
          v-for="a in alignOptions"
          :key="a.value"
          class="image-toolbar__btn"
          :class="{ 'image-toolbar__btn--active': alignment === a.value }"
          :title="a.title"
          @click="setAlign(a.value)"
          v-html="a.icon"
        />
      </div>

      <div class="image-resizer">
        <div class="resize-handle resize-handle--left" @mousedown.prevent="startResize($event, 'left')" />
        <img
          :src="node.attrs.src"
          :alt="node.attrs.alt || ''"
          :width="currentWidth || undefined"
          draggable="false"
          class="image-resizer__img"
        />
        <div class="resize-handle resize-handle--right" @mousedown.prevent="startResize($event, 'right')" />
      </div>
    </div>
  </NodeViewWrapper>
</template>

<script setup lang="ts">
import { NodeViewWrapper, nodeViewProps } from '@tiptap/vue-3'

const props = defineProps(nodeViewProps)

const hovered = ref(false)
const currentWidth = ref<number>(props.node.attrs.width || 0)
const alignment = ref<string>(props.node.attrs.align || 'center')

const alignOptions = [
  {
    value: 'left',
    title: '左对齐',
    icon: `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path d="M4 2C4 1.44772 3.55228 1 3 1C2.44772 1 2 1.44772 2 2V22C2 22.5523 2.44772 23 3 23C3.55228 23 4 22.5523 4 22V2Z"/><path fill-rule="evenodd" clip-rule="evenodd" d="M10 4C8.34315 4 7 5.34315 7 7V17C7 18.6569 8.34315 20 10 20H19C20.6569 20 22 18.6569 22 17V7C22 5.34315 20.6569 4 19 4H10ZM9 7C9 6.44772 9.44772 6 10 6H19C19.5523 6 20 6.44772 20 7V17C20 17.5523 19.5523 18 19 18H10C9.44772 18 9 17.5523 9 17V7Z"/></svg>`,
  },
  {
    value: 'center',
    title: '居中',
    icon: `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path d="M12 1C12.5523 1 13 1.44772 13 2V22C13 22.5523 12.5523 23 12 23C11.4477 23 11 22.5523 11 22V2C11 1.44772 11.4477 1 12 1Z"/><path d="M2 7C2 5.34315 3.34315 4 5 4H7C7.55228 4 8 4.44772 8 5C8 5.55228 7.55228 6 7 6H5C4.44772 6 4 6.44772 4 7V17C4 17.5523 4.44772 18 5 18H7C7.55228 18 8 18.4477 8 19C8 19.5523 7.55228 20 7 20H5C3.34315 20 2 18.6569 2 17V7Z"/><path d="M19 4C20.6569 4 22 5.34315 22 7V17C22 18.6569 20.6569 20 19 20H17C16.4477 20 16 19.5523 16 19C16 18.4477 16.4477 18 17 18H19C19.5523 18 20 17.5523 20 17V7C20 6.44772 19.5523 6 19 6H17C16.4477 6 16 5.55228 16 5C16 4.44772 16.4477 4 17 4H19Z"/></svg>`,
  },
  {
    value: 'right',
    title: '右对齐',
    icon: `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" clip-rule="evenodd" d="M21 1C21.5523 1 22 1.44772 22 2V22C22 22.5523 21.5523 23 21 23C20.4477 23 20 22.5523 20 22V2C20 1.44772 20.4477 1 21 1Z"/><path fill-rule="evenodd" clip-rule="evenodd" d="M2 7C2 5.34315 3.34315 4 5 4H14C15.6569 4 17 5.34315 17 7V17C17 18.6569 15.6569 20 14 20H5C3.34315 20 2 18.6569 2 17V7ZM5 6C4.44772 6 4 6.44772 4 7V17C4 17.5523 4.44772 18 5 18H14C14.5523 18 15 17.5523 15 17V7C15 6.44772 14.5523 6 14 6H5Z"/></svg>`,
  },
]

const setAlign = (value: string) => {
  alignment.value = value
  props.updateAttributes({ align: value })
}

const startResize = (event: MouseEvent, _side: string) => {
  const startX = event.clientX
  const img = (event.target as HTMLElement).parentElement?.querySelector('img')
  if (!img) return
  const startWidth = img.offsetWidth

  const onMouseMove = (e: MouseEvent) => {
    const diff = e.clientX - startX
    const newWidth = Math.max(100, _side === 'right' ? startWidth + diff : startWidth - diff)
    currentWidth.value = newWidth
  }

  const onMouseUp = () => {
    document.removeEventListener('mousemove', onMouseMove)
    document.removeEventListener('mouseup', onMouseUp)
    props.updateAttributes({ width: currentWidth.value })
  }

  document.addEventListener('mousemove', onMouseMove)
  document.addEventListener('mouseup', onMouseUp)
}
</script>
