<template>
  <node-view-wrapper>
    <div :id="uid" class="code-group" data-code-group>
      <div class="code-group__tabs" contenteditable="false">
        <button
          v-for="tab in tabs"
          :key="tab.index"
          type="button"
          class="code-group__tab"
          tabindex="-1"
          :class="{ 'is-active': tab.index === active }"
          @click="setActive(tab.index)"
        >
          <span class="code-group__tab-lang">{{ tab.lang.toUpperCase() }}</span>
          <span class="code-group__tab-title">{{ tab.title }}</span>
        </button>
      </div>
      <div class="code-group__content">
        <node-view-content />
      </div>
    </div>
  </node-view-wrapper>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { NodeViewWrapper, NodeViewContent, nodeViewProps } from '@tiptap/vue-3'

const props = defineProps(nodeViewProps)
const active = ref(0)

let _counter = 0
const uid = `cg-${Date.now()}-${_counter++}`

const tabs = computed(() => {
  const items: Array<{ index: number; lang: string; title: string }> = []
  let idx = 0
  props.node.content.forEach((child) => {
    if (child.type.name !== 'codeBlock') return
    const lang = child.attrs.language || 'text'
    const title = child.attrs.title || lang
    items.push({ index: idx, lang, title })
    idx += 1
  })
  return items
})

// Inject a <style> into <head> scoped by our unique id.
// CSS rules in <head> can't be reset by ProseMirror DOM re-renders.
const styleEl = document.createElement('style')
document.head.appendChild(styleEl)

const updateStyle = () => {
  const rules: string[] = []
  const total = tabs.value.length || 1
  for (let i = 1; i <= total; i++) {
    const display = i === active.value + 1 ? 'block' : 'none'
    rules.push(`#${uid} .code-group__content pre:nth-of-type(${i}) { display: ${display} !important; }`)
  }
  styleEl.textContent = rules.join('\n')
}

const setActive = (i: number) => {
  active.value = i
  updateStyle()
}

watch(tabs, () => {
  if (active.value >= tabs.value.length) active.value = 0
  updateStyle()
}, { immediate: true })

onBeforeUnmount(() => {
  styleEl.remove()
})
</script>
