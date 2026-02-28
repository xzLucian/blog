<template>
  <NodeViewWrapper class="image-upload-wrapper">
    <div v-if="!imgSrc" class="upload-zone" @click="triggerUpload" @dragover.prevent @drop.prevent="onDrop" @paste="onPaste">
      <input ref="fileInput" type="file" accept="image/*" hidden @change="onFileChange" />
      <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16v-8"/><path d="m8 12 4-4 4 4"/><path d="M20 16.7A4.5 4.5 0 0 0 17.5 8h-1.8A7 7 0 1 0 4 14.9"/></svg>
      <span class="upload-text">Click to upload image</span>
      <span class="upload-hint">support paste image</span>
      <div class="url-row" @click.stop>
        <input v-model="urlInput" class="url-input" placeholder="or paste image URL..." @keydown.enter="confirmUrl" />
        <button class="url-confirm" @click="confirmUrl">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path fill-rule="evenodd" clip-rule="evenodd" d="M21 4C21 3.44772 20.5523 3 20 3C19.4477 3 19 3.44772 19 4V11C19 11.7956 18.6839 12.5587 18.1213 13.1213C17.5587 13.6839 16.7956 14 16 14H6.41421L9.70711 10.7071C10.0976 10.3166 10.0976 9.68342 9.70711 9.29289C9.31658 8.90237 8.68342 8.90237 8.29289 9.29289L3.29289 14.2929C2.90237 14.6834 2.90237 15.3166 3.29289 15.7071L8.29289 20.7071C8.68342 21.0976 9.31658 21.0976 9.70711 20.7071C10.0976 20.3166 10.0976 19.6834 9.70711 19.2929L6.41421 16H16C17.3261 16 18.5979 15.4732 19.5355 14.5355C20.4732 13.5979 21 12.3261 21 11V4Z"/></svg>
        </button>
      </div>
    </div>
    <div v-else class="image-preview">
      <img :src="imgSrc" />
    </div>
    <button class="remove-btn" @click="deleteNode" title="删除">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path fill-rule="evenodd" clip-rule="evenodd" d="M7 5V4C7 3.17477 7.40255 2.43324 7.91789 1.91789C8.43324 1.40255 9.17477 1 10 1H14C14.8252 1 15.5668 1.40255 16.0821 1.91789C16.5975 2.43324 17 3.17477 17 4V5H21C21.5523 5 22 5.44772 22 6C22 6.55228 21.5523 7 21 7H20V20C20 20.8252 19.5975 21.5668 19.0821 22.0821C18.5668 22.5975 17.8252 23 17 23H7C6.17477 23 5.43324 22.5975 4.91789 22.0821C4.40255 21.5668 4 20.8252 4 20V7H3C2.44772 7 2 6.55228 2 6C2 5.44772 2.44772 5 3 5H7ZM9 4C9 3.82523 9.09745 3.56676 9.33211 3.33211C9.56676 3.09745 9.82523 3 10 3H14C14.1748 3 14.4332 3.09745 14.6679 3.33211C14.9025 3.56676 15 3.82523 15 4V5H9V4ZM6 7V20C6 20.1748 6.09745 20.4332 6.33211 20.6679C6.56676 20.9025 6.82523 21 7 21H17C17.1748 21 17.4332 20.9025 17.6679 20.6679C17.9025 20.4332 18 20.1748 18 20V7H6Z"/></svg>
    </button>
  </NodeViewWrapper>
</template>

<script setup lang="ts">
import { NodeViewWrapper, nodeViewProps } from '@tiptap/vue-3'
const props = defineProps(nodeViewProps)
const fileInput = ref<HTMLInputElement>()
const urlInput = ref('')
const imgSrc = ref(props.node.attrs.src || '')

const triggerUpload = () => fileInput.value?.click()

const toBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

const uploadFile = async (file: File) => {
  try {
    const base64 = await toBase64(file)
    replaceWithImage(base64)
  } catch {
    ElMessage.error('转换失败')
  }
}

const replaceWithImage = (src: string) => {
  props.editor.chain().focus().deleteRange({ from: props.getPos(), to: props.getPos() + props.node.nodeSize }).setImage({ src }).run()
}

const onFileChange = (e: Event) => {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (file) uploadFile(file)
}

const onDrop = (e: DragEvent) => {
  const file = e.dataTransfer?.files[0]
  if (file?.type.startsWith('image/')) uploadFile(file)
}

const onPaste = (e: ClipboardEvent) => {
  const file = Array.from(e.clipboardData?.items || []).find(i => i.type.startsWith('image/'))?.getAsFile()
  if (file) uploadFile(file)
}

const confirmUrl = () => {
  if (urlInput.value.trim()) replaceWithImage(urlInput.value.trim())
}
</script>
