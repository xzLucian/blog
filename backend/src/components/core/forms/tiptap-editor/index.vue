<template>
  <div class="tiptap-editor" :class="{ 'tiptap-editor--flex': $slots['before-content'] }">
    <div class="tiptap-toolbar" v-if="editor">
      <!-- Undo / Redo -->
      <button @click="editor.chain().focus().undo().run()" :disabled="!editor.can().undo()" v-html="icons.undo" title="撤销" />
      <button @click="editor.chain().focus().redo().run()" :disabled="!editor.can().redo()" v-html="icons.redo" title="重做" />
      <span class="divider" />

      <!-- Headings Dropdown -->
      <div class="toolbar-dropdown" ref="headingDropdownRef">
        <button
          class="dropdown-trigger"
          :class="{ active: editor.isActive('heading') }"
          @click="headingOpen = !headingOpen"
          title="标题"
        >
          <span v-html="icons.heading" />
          <span v-html="icons.dropdownArrow" class="dropdown-arrow" />
        </button>
        <div class="dropdown-menu" v-show="headingOpen">
          <button @click="selectHeading(0)" :class="{ active: !editor.isActive('heading') }">正文</button>
          <button v-for="l in 4" :key="l" @click="selectHeading(l as Level)" :class="{ active: editor.isActive('heading', { level: l }) }">
            <span v-html="icons[`h${l}` as keyof typeof icons]" />
          </button>
        </div>
      </div>
      <span class="divider" />

      <!-- Lists Dropdown -->
      <div class="toolbar-dropdown" ref="listDropdownRef">
        <button
          class="dropdown-trigger"
          :class="{ active: editor.isActive('bulletList') || editor.isActive('orderedList') || editor.isActive('taskList') }"
          @click="listOpen = !listOpen"
          title="列表"
        >
          <span v-html="icons.bulletList" />
          <span v-html="icons.dropdownArrow" class="dropdown-arrow" />
        </button>
        <div class="dropdown-menu" v-show="listOpen">
          <button @click="selectList('bulletList')" :class="{ active: editor.isActive('bulletList') }">
            <span v-html="icons.bulletList" /> Bullet List
          </button>
          <button @click="selectList('orderedList')" :class="{ active: editor.isActive('orderedList') }">
            <span v-html="icons.orderedList" /> Ordered List
          </button>
          <button @click="selectList('taskList')" :class="{ active: editor.isActive('taskList') }">
            <span v-html="icons.taskList" /> Task List
          </button>
        </div>
      </div>
      <span class="divider" />

      <!-- Blockquote / CodeBlock -->
      <button @click="editor.chain().focus().toggleBlockquote().run()" :class="{ active: editor.isActive('blockquote') }" v-html="icons.blockquote" title="引用" />
      <button @click="editor.chain().focus().toggleCodeBlock().run()" :class="{ active: editor.isActive('codeBlock') }" v-html="icons.codeBlock" title="代码块" />
      <span class="divider" />

      <!-- Bold / Italic / Strike / Code / Underline / Highlight -->
      <button @click="editor.chain().focus().toggleBold().run()" :class="{ active: editor.isActive('bold') }" v-html="icons.bold" title="加粗" />
      <button @click="editor.chain().focus().toggleItalic().run()" :class="{ active: editor.isActive('italic') }" v-html="icons.italic" title="斜体" />
      <button @click="editor.chain().focus().toggleStrike().run()" :class="{ active: editor.isActive('strike') }" v-html="icons.strike" title="删除线" />
      <button @click="editor.chain().focus().toggleCode().run()" :class="{ active: editor.isActive('code') }" v-html="icons.code" title="行内代码" />
      <button @click="editor.chain().focus().toggleUnderline().run()" :class="{ active: editor.isActive('underline') }" v-html="icons.underline" title="下划线" />
      <button @click="editor.chain().focus().toggleHighlight().run()" :class="{ active: editor.isActive('highlight') }" v-html="icons.highlight" title="高亮" />
      <span class="divider" />

      <!-- Link -->
      <div class="toolbar-dropdown" ref="linkDropdownRef">
        <button @click="toggleLinkBar" :class="{ active: editor.isActive('link') }" v-html="icons.link" title="链接" />
        <div class="link-popover" v-show="linkBarVisible">
          <input
            ref="linkInputRef"
            v-model="linkUrl"
            class="link-input"
            placeholder="Paste a link..."
            @keydown.enter="confirmLink"
          />
          <button class="link-action" @click="confirmLink" title="确认">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" clip-rule="evenodd" d="M21 4C21 3.44772 20.5523 3 20 3C19.4477 3 19 3.44772 19 4V11C19 11.7956 18.6839 12.5587 18.1213 13.1213C17.5587 13.6839 16.7956 14 16 14H6.41421L9.70711 10.7071C10.0976 10.3166 10.0976 9.68342 9.70711 9.29289C9.31658 8.90237 8.68342 8.90237 8.29289 9.29289L3.29289 14.2929C2.90237 14.6834 2.90237 15.3166 3.29289 15.7071L8.29289 20.7071C8.68342 21.0976 9.31658 21.0976 9.70711 20.7071C10.0976 20.3166 10.0976 19.6834 9.70711 19.2929L6.41421 16H16C17.3261 16 18.5979 15.4732 19.5355 14.5355C20.4732 13.5979 21 12.3261 21 11V4Z" fill="currentColor"></path></svg>
          </button>
          <button class="link-action" @click="openLink" title="打开链接" :disabled="!linkUrl">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path d="M14 3C14 2.44772 14.4477 2 15 2H21C21.5523 2 22 2.44772 22 3V9C22 9.55228 21.5523 10 21 10C20.4477 10 20 9.55228 20 9V5.41421L10.7071 14.7071C10.3166 15.0976 9.68342 15.0976 9.29289 14.7071C8.90237 14.3166 8.90237 13.6834 9.29289 13.2929L18.5858 4H15C14.4477 4 14 3.55228 14 3Z" fill="currentColor"></path><path d="M4.29289 7.29289C4.48043 7.10536 4.73478 7 5 7H11C11.5523 7 12 6.55228 12 6C12 5.44772 11.5523 5 11 5H5C4.20435 5 3.44129 5.31607 2.87868 5.87868C2.31607 6.44129 2 7.20435 2 8V19C2 19.7957 2.31607 20.5587 2.87868 21.1213C3.44129 21.6839 4.20435 22 5 22H16C16.7957 22 17.5587 21.6839 18.1213 21.1213C18.6839 20.5587 19 19.7957 19 19V13C19 12.4477 18.5523 12 18 12C17.4477 12 17 12.4477 17 13V19C17 19.2652 16.8946 19.5196 16.7071 19.7071C16.5196 19.8946 16.2652 20 16 20H5C4.73478 20 4.48043 19.8946 4.29289 19.7071C4.10536 19.5196 4 19.2652 4 19V8C4 7.73478 4.10536 7.48043 4.29289 7.29289Z" fill="currentColor"></path></svg>
          </button>
          <button class="link-action danger" @click="removeLink" title="删除链接">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" clip-rule="evenodd" d="M7 5V4C7 3.17477 7.40255 2.43324 7.91789 1.91789C8.43324 1.40255 9.17477 1 10 1H14C14.8252 1 15.5668 1.40255 16.0821 1.91789C16.5975 2.43324 17 3.17477 17 4V5H21C21.5523 5 22 5.44772 22 6C22 6.55228 21.5523 7 21 7H20V20C20 20.8252 19.5975 21.5668 19.0821 22.0821C18.5668 22.5975 17.8252 23 17 23H7C6.17477 23 5.43324 22.5975 4.91789 22.0821C4.40255 21.5668 4 20.8252 4 20V7H3C2.44772 7 2 6.55228 2 6C2 5.44772 2.44772 5 3 5H7ZM9 4C9 3.82523 9.09745 3.56676 9.33211 3.33211C9.56676 3.09745 9.82523 3 10 3H14C14.1748 3 14.4332 3.09745 14.6679 3.33211C14.9025 3.56676 15 3.82523 15 4V5H9V4ZM6 7V20C6 20.1748 6.09745 20.4332 6.33211 20.6679C6.56676 20.9025 6.82523 21 7 21H17C17.1748 21 17.4332 20.9025 17.6679 20.6679C17.9025 20.4332 18 20.1748 18 20V7H6Z" fill="currentColor"></path></svg>
          </button>
        </div>
      </div>
      <span class="divider" />

      <!-- Superscript / Subscript -->
      <button @click="editor.chain().focus().toggleSuperscript().run()" :class="{ active: editor.isActive('superscript') }" v-html="icons.superscript" title="上标" />
      <button @click="editor.chain().focus().toggleSubscript().run()" :class="{ active: editor.isActive('subscript') }" v-html="icons.subscript" title="下标" />
      <span class="divider" />

      <!-- Align -->
      <button @click="editor.chain().focus().setTextAlign('left').run()" :class="{ active: editor.isActive({ textAlign: 'left' }) }" v-html="icons.alignLeft" title="左对齐" />
      <button @click="editor.chain().focus().setTextAlign('right').run()" :class="{ active: editor.isActive({ textAlign: 'right' }) }" v-html="icons.alignRight" title="右对齐" />
      <button @click="editor.chain().focus().setTextAlign('center').run()" :class="{ active: editor.isActive({ textAlign: 'center' }) }" v-html="icons.alignCenter" title="居中" />
      <button @click="editor.chain().focus().setTextAlign('justify').run()" :class="{ active: editor.isActive({ textAlign: 'justify' }) }" v-html="icons.alignJustify" title="两端对齐" />
      <span class="divider" />

      <!-- Image -->
      <button @click="addImage" class="add-image-btn" title="插入图片">
        <span v-html="icons.image" /> Add
      </button>
      <slot name="toolbar-extra" />
    </div>
    <slot name="before-content" />
    <EditorContent :editor="editor" class="tiptap-content" />
  </div>
</template>

<script setup lang="ts">
  import { useEditor, EditorContent } from '@tiptap/vue-3'
  import StarterKit from '@tiptap/starter-kit'
  import Underline from '@tiptap/extension-underline'
  import Link from '@tiptap/extension-link'
  import { ImageResize } from './image-resize'
  import TextAlign from '@tiptap/extension-text-align'
  import Highlight from '@tiptap/extension-highlight'
  import TaskList from '@tiptap/extension-task-list'
  import TaskItem from '@tiptap/extension-task-item'
  import Placeholder from '@tiptap/extension-placeholder'
  import Superscript from '@tiptap/extension-superscript'
  import Subscript from '@tiptap/extension-subscript'
  import CodeBlock from '@tiptap/extension-code-block'
  import { mergeAttributes, textblockTypeInputRule } from '@tiptap/core'
  import { ShikiPlugin } from './shiki-plugin'
  import { icons } from './icons'
  import { ImageUpload } from './image-upload'
  import { Callout } from './callout'
  import { CodeGroup } from './code-group'

  type Level = 1 | 2 | 3 | 4

  defineOptions({ name: 'TiptapEditor' })

  interface Props {
    height?: string
    placeholder?: string
  }

  const props = withDefaults(defineProps<Props>(), {
    height: '500px',
    placeholder: '请输入内容...'
  })

  const modelValue = defineModel<string>({ required: true })

  const backtickInputRegexWithTitle = new RegExp('^```([\\w-]+)?(?:\\s*\\[([^\\]]+)\\])?[\\s\\n]$')
  const tildeInputRegexWithTitle = new RegExp('^~~~([\\w-]+)?(?:\\s*\\[([^\\]]+)\\])?[\\s\\n]$')

  const headingOpen = ref(false)
  const headingDropdownRef = ref<HTMLElement>()
  const listOpen = ref(false)
  const listDropdownRef = ref<HTMLElement>()
  const linkDropdownRef = ref<HTMLElement>()

  const selectHeading = (level: Level | 0) => {
    if (level === 0) {
      editor.value?.chain().focus().setParagraph().run()
    } else {
      editor.value?.chain().focus().toggleHeading({ level }).run()
    }
    headingOpen.value = false
  }

  const selectList = (type: 'bulletList' | 'orderedList' | 'taskList') => {
    const chain = editor.value?.chain().focus()
    if (type === 'bulletList') chain?.toggleBulletList().run()
    else if (type === 'orderedList') chain?.toggleOrderedList().run()
    else chain?.toggleTaskList().run()
    listOpen.value = false
  }

  const onClickOutside = (e: MouseEvent) => {
    const target = e.target as Node
    if (headingDropdownRef.value && !headingDropdownRef.value.contains(target)) {
      headingOpen.value = false
    }
    if (listDropdownRef.value && !listDropdownRef.value.contains(target)) {
      listOpen.value = false
    }
    if (linkDropdownRef.value && !linkDropdownRef.value.contains(target)) {
      linkBarVisible.value = false
    }
  }

  onMounted(() => document.addEventListener('click', onClickOutside))
  onBeforeUnmount(() => document.removeEventListener('click', onClickOutside))

  const editor = useEditor({
    content: modelValue.value,
    extensions: [
      StarterKit.configure({ codeBlock: false }),
      Underline,
      Link.configure({ openOnClick: false }),
      ImageResize,
      TextAlign.configure({ types: ['heading', 'paragraph'] }),
      Highlight,
      TaskList,
      TaskItem.configure({ nested: true }),
      Placeholder.configure({ placeholder: props.placeholder }),
      Superscript,
      Subscript,
      CodeBlock.extend({
        addAttributes() {
          return {
            ...this.parent?.(),
            title: {
              default: null,
              parseHTML: (element) => (element as HTMLElement).getAttribute('data-title'),
            },
          }
        },
        addKeyboardShortcuts() {
          return {
            Enter: () => {
              if (!this.editor.isActive('codeBlock')) return false
              const { state } = this.editor
              const { $from, empty } = state.selection
              if (!empty) return false
              if ($from.parent.type.name !== 'codeBlock') return false

              const text = $from.parent.textContent
              const offset = $from.parentOffset
              const after = text.slice(offset)
              if (after.length > 0) return false

              const lineStart = text.lastIndexOf('\n', offset - 1) + 1
              const lineContent = text.slice(lineStart, offset)
              if (lineContent.length > 0) return false

              if (offset === 0 || text[offset - 1] !== '\n') return false
              return this.editor.commands.exitCode()
            },
          }
        },
        addInputRules() {
          return [
            textblockTypeInputRule({
              find: backtickInputRegexWithTitle,
              type: this.type,
              getAttributes: (match) => ({
                language: match[1],
                title: match[2] || null,
              }),
            }),
            textblockTypeInputRule({
              find: tildeInputRegexWithTitle,
              type: this.type,
              getAttributes: (match) => ({
                language: match[1],
                title: match[2] || null,
              }),
            }),
          ]
        },
        addProseMirrorPlugins() {
          return [...(this.parent?.() || []), ShikiPlugin(this.name)]
        },
        renderHTML({ node, HTMLAttributes }) {
          const lang = node.attrs.language || ''
          const dataAttrs: Record<string, string> = {}
          if (node.attrs.title) dataAttrs['data-title'] = node.attrs.title
          return [
            'pre',
            mergeAttributes(this.options.HTMLAttributes, HTMLAttributes, {
              class: 'shiki',
              'data-lang': lang || 'text',
              ...dataAttrs,
            }),
            [
              'code',
              {
                class: lang ? this.options.languageClassPrefix + lang : null,
              },
              0,
            ],
          ]
        },
      }),
      CodeGroup,
      ImageUpload,
      Callout
    ],
    editorProps: {
      handleDOMEvents: {
        keydown(view, event) {
          if (event.key !== 'Tab') return false
          const { state } = view
          const { $from } = state.selection
          event.preventDefault()
          // Keep focus in the editor and insert two-space indent.
          view.dispatch(state.tr.insertText('　　'))
          return true
        },
      },
    },
    onUpdate: ({ editor: e }) => {
      modelValue.value = e.getHTML()
    }
  })

  // 监听外部 modelValue 变化（编辑模式回填）
  watch(modelValue, (val) => {
    if (editor.value && val !== editor.value.getHTML()) {
      editor.value.commands.setContent(val, { emitUpdate: false })
    }
  })

  const linkBarVisible = ref(false)
  const linkUrl = ref('')
  const linkInputRef = ref<HTMLInputElement>()

  const toggleLinkBar = () => {
    if (linkBarVisible.value) {
      linkBarVisible.value = false
      return
    }
    linkUrl.value = editor.value?.getAttributes('link').href || ''
    linkBarVisible.value = true
    nextTick(() => linkInputRef.value?.focus())
  }

  const confirmLink = () => {
    if (linkUrl.value) {
      editor.value?.chain().focus().extendMarkRange('link').setLink({ href: linkUrl.value }).run()
    } else {
      editor.value?.chain().focus().extendMarkRange('link').unsetLink().run()
    }
    linkBarVisible.value = false
  }

  const openLink = () => {
    if (linkUrl.value) window.open(linkUrl.value, '_blank')
  }

  const removeLink = () => {
    editor.value?.chain().focus().extendMarkRange('link').unsetLink().run()
    linkBarVisible.value = false
  }

  const addImage = () => {
    ;(editor.value?.chain().focus() as any).insertImageUpload().run()
  }

  const wordCount = computed(() => editor.value?.getText().length ?? 0)

  defineExpose({ wordCount })

  onBeforeUnmount(() => {
    editor.value?.destroy()
  })
</script>

<style lang="scss" scoped>
  $radius: calc(var(--custom-radius) / 3 + 2px);

  .tiptap-editor--flex {
    display: flex;
    flex-direction: column;
    flex: 1;
    overflow: hidden;

    :deep(.tiptap-content) {
      flex: 1;
      overflow-y: auto;
    }
  }

  .tiptap-toolbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 2px;
    padding: 6px 8px;
    border-bottom: 1px solid var(--art-gray-200);
    background: var(--art-gray-50);

    button {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 28px;
      height: 28px;
      padding: 0;
      border: none;
      border-radius: 8px;
      background: transparent;
      color: var(--art-gray-800);
      cursor: pointer;
      transition: background 0.15s;

      &:hover {
        background: var(--art-gray-200);
      }

      &.active {
        background: var(--art-gray-300);
        color: var(--el-color-primary);
      }

      &:disabled {
        opacity: 0.35;
        cursor: not-allowed;
      }

      :deep(svg) {
        width: 15px;
        height: 15px;
        stroke-width: 0.5px;
      }
    }

    .add-image-btn {
      width: auto;
      gap: 4px;
      padding: 0 8px;
      font-size: 13px;
    }

    .divider {
      width: 1px;
      height: 20px;
      margin: 0 4px;
      background: var(--art-gray-300);
    }

    .toolbar-dropdown {
      position: relative;

      .dropdown-trigger {
        display: flex;
        align-items: center;
        width: auto;
        gap: 0;
        padding: 0 4px;

        .dropdown-arrow :deep(svg) {
          width: 14px;
          height: 14px;
        }
      }

      .dropdown-menu {
        position: absolute;
        top: 100%;
        left: 0;
        z-index: 10;
        display: flex;
        flex-direction: column;
        min-width: 140px;
        padding: 4px;
        margin-top: 4px;
        background: var(--el-bg-color);
        border-radius: 8px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.12);

        button {
          justify-content: flex-start;
          width: 100%;
          height: 34px;
          padding: 0 8px;
          font-size: 13px;
          gap: 8px;
          border-radius: 6px;
        }
      }
    }
  }

  .link-popover {
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    z-index: 10;
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 6px 10px;
    margin-top: 4px;
    min-width: 320px;
    background: var(--el-bg-color);
    border-radius: 10px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
  }

  .link-input {
    flex: 1;
    border: none;
    outline: none;
    background: transparent;
    font-size: 13px;
    color: var(--art-gray-800);

    &::placeholder {
      color: var(--art-gray-400);
    }
  }

  .link-action {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    padding: 0;
    border: none;
    border-radius: 6px;
    background: transparent;
    color: var(--art-gray-600);
    cursor: pointer;

    &:hover {
      background: var(--art-gray-200);
    }

    &:disabled {
      opacity: 0.35;
      cursor: not-allowed;
    }

    &.danger:hover {
      color: var(--el-color-danger);
    }
  }
</style>

<style lang="scss">
  .tiptap-content {
    .tiptap {
      padding: 12px 16px;
      min-height: v-bind(height);
      outline: none;
      font-family: 'DM Sans', sans-serif;

      > * + * {
        margin-top: 0.5em;
      }

      h1, h2, h3, h4 {
        font-weight: 600;
        line-height: 1.3;
      }

      h1 { font-size: 2em; }
      h2 { font-size: 1.5em; }
      h3 { font-size: 1.25em; }
      h4 { font-size: 1.1em; }

      strong, b { font-weight: 600; }
      em, i { font-style: italic; }

      ul { list-style: disc; padding-left: 1.5em; }
      ol { list-style: decimal; padding-left: 1.5em; }

      ul[data-type="taskList"] {
        list-style: none;
        padding-left: 0;

        li {
          display: flex;
          align-items: flex-start;
          gap: 8px;

          label { margin-top: 3px; }

          > div { flex: 1; }
        }
      }

      blockquote {
        padding: 8px 16px;
        border-left: 4px solid var(--art-gray-300);
        background: var(--art-gray-100);
        color: var(--art-gray-700);
      }

      pre {
        padding: 12px 16px;
        border-radius: 6px;
        border: 1px solid var(--art-gray-200);
        background: var(--shiki-light-bg, #0b0b0b) !important;
        overflow-x: auto;
        position: relative;

        code {
          font-family: 'Fira Code', 'Consolas', monospace;
          font-size: 0.9em;
          background: none;
        }
      }

      pre.shiki {
        --shiki-light-bg: #0b0b0b;
        --shiki-dark-bg: #0b0b0b;
        border-color: #1f2328;
      }

      pre.shiki span {
        color: var(--shiki-dark, inherit) !important;
      }

      .code-group {
        margin: 12px 0;
        border: 1px solid #1f2328;
        border-radius: 8px;
        background: #0b0b0b;
      }

      .code-group__tabs {
        display: flex;
        align-items: center;
        gap: 24px;
        padding: 0 6px;
        border-bottom: 1px solid #1f2328;
        background: transparent;
      }

      .code-group__tab {
        position: relative;
        display: inline-flex;
        align-items: center;
        gap: 12px;
        padding: 12px 2px;
        border: none;
        background: transparent;
        color: #6b7280;
        font-size: 16px;
        font-weight: 500;
        cursor: pointer;
      }

      .code-group__tab.is-active {
        color: #111827;
      }

      .code-group__tab.is-active::after {
        content: '';
        position: absolute;
        left: 0;
        right: 0;
        bottom: -1px;
        height: 2px;
        background: #3b5bdb;
        border-radius: 2px;
      }

      .code-group__tab-lang {
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 0.02em;
        color: #2563eb;
      }

      .code-group__tab-title {
        color: #e5e7eb;
      }

      .code-group__content {
        padding: 10px 0;
      }

      .code-group__content pre {
        margin: 0;
        border: none;
        border-radius: 0;
      }


      .code-block-toolbar {
        position: absolute;
        top: 8px;
        right: 10px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: #e5e7eb;
        font-size: 12px;
        opacity: 0.55;
        transition: opacity 0.15s;
        pointer-events: none;
      }

      pre.shiki:hover .code-block-toolbar {
        opacity: 1;
        pointer-events: auto;
      }

      .code-block-copy {
        border: none;
        background: transparent;
        padding: 0;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        color: inherit;
        cursor: pointer;
      }

      .code-block-copy-label {
        display: inline;
        text-transform: lowercase;
      }

      .code-block-copy-icon {
        width: 16px;
        height: 16px;
        stroke: currentColor;
        fill: none;
        stroke-width: 1.6;
        display: none;
      }

      pre.shiki:hover .code-block-copy-label {
        display: none;
      }

      pre.shiki:hover .code-block-copy-icon {
        display: inline-flex;
      }

      code {
        padding: 2px 6px;
        border-radius: 4px;
        background: var(--art-gray-200);
        font-size: 0.9em;
      }

      a {
        color: var(--el-color-primary);
        text-decoration: underline;
        cursor: pointer;
      }

      img {
        max-width: 100%;
        height: auto;
        border-radius: 4px;
      }

      mark {
        background: #ffff00;
        padding: 0 2px;
      }

      hr {
        border: none;
        border-top: 1px solid var(--art-gray-300);
        margin: 1em 0;
      }

    }

    .tiptap p.is-editor-empty:first-child::before {
      content: attr(data-placeholder);
      float: left;
      color: var(--art-gray-400);
      pointer-events: none;
      height: 0;
    }
  }

  html.dark .tiptap-content .tiptap pre.shiki {
    background: var(--shiki-dark-bg, #0b0b0b) !important;
  }

  html.dark .tiptap-content .tiptap pre.shiki span {
    color: var(--shiki-dark, inherit) !important;
  }

  html.dark .tiptap-content .tiptap .code-block-toolbar,
  html.dark .tiptap-content .tiptap .code-block-copy {
    color: var(--art-gray-100);
  }

  html.dark .tiptap-content .tiptap .code-group {
    background: transparent;
    border-color: #1f2328;
  }

  html.dark .tiptap-content .tiptap .code-group__tabs {
    border-bottom-color: #1f2328;
  }

  html.dark .tiptap-content .tiptap .code-group__tab {
    color: #9ca3af;
  }

  html.dark .tiptap-content .tiptap .code-group__tab.is-active {
    color: #f3f4f6;
  }

  html.dark .tiptap-content .tiptap .code-group__tab-lang {
    color: #60a5fa;
  }

  .image-upload-wrapper {
    position: relative;
    margin: 8px 0;
  }

  .upload-zone {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    padding: 24px 16px 12px;
    border: 2px dashed var(--art-gray-300);
    border-radius: 8px;
    background: var(--art-gray-50);
    cursor: pointer;
    color: var(--art-gray-400);
    transition: border-color 0.2s;

    &:hover {
      border-color: var(--el-color-primary);
    }
  }

  .upload-text {
    font-size: 13px;
    font-weight: 500;
    color: var(--art-gray-600);
  }

  .upload-hint {
    font-size: 11px;
    color: var(--art-gray-400);
  }

  .url-row {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-top: 6px;
    width: 100%;
    max-width: 320px;
  }

  .url-input {
    flex: 1;
    height: 30px;
    padding: 0 8px;
    border: 1px solid var(--art-gray-300);
    border-radius: 6px;
    outline: none;
    font-size: 12px;
    background: var(--el-bg-color);

    &:focus {
      border-color: var(--el-color-primary);
    }
  }

  .url-confirm {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    border: none;
    border-radius: 6px;
    background: transparent;
    color: var(--art-gray-600);
    cursor: pointer;

    &:hover {
      background: var(--art-gray-200);
    }
  }

  .image-upload-wrapper .remove-btn {
    position: absolute;
    top: 4px;
    right: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border: none;
    border-radius: 6px;
    background: transparent;
    color: var(--art-gray-400);
    cursor: pointer;

    &:hover {
      background: var(--art-gray-200);
      color: var(--el-color-danger);
    }
  }

  .image-preview img {
    max-width: 100%;
    border-radius: 6px;
  }

  /* Image resize & align */
  .image-view {
    margin: 8px 0;
  }

  .image-align-left {
    display: flex;
    justify-content: flex-start;
  }

  .image-align-center {
    display: flex;
    justify-content: center;
  }

  .image-align-right {
    display: flex;
    justify-content: flex-end;
  }

  .image-container {
    position: relative;
    display: inline-block;
    max-width: 100%;
    line-height: 0;
  }

  .image-container--selected {
    outline: 2px solid var(--el-color-primary);
    border-radius: 4px;
  }

  .image-toolbar {
    position: absolute;
    top: -40px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 10;
    display: flex;
    gap: 2px;
    padding: 4px;
    background: var(--el-bg-color);
    border-radius: 8px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.12);
  }

  .image-toolbar__btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    padding: 0;
    border: none;
    border-radius: 6px;
    background: transparent;
    color: var(--art-gray-600);
    cursor: pointer;
    transition: background 0.15s;

    &:hover {
      background: var(--art-gray-200);
    }

    &--active {
      background: var(--art-gray-300);
      color: var(--el-color-primary);
    }
  }

  .image-resizer {
    position: relative;
    display: inline-flex;
    align-items: stretch;
  }

  .image-resizer__img {
    max-width: 100%;
    height: auto;
    border-radius: 4px;
    display: block;
  }

  .resize-handle {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 6px;
    cursor: col-resize;
    opacity: 0;
    transition: opacity 0.15s;
    z-index: 5;

    &::after {
      content: '';
      position: absolute;
      top: 50%;
      transform: translateY(-50%);
      width: 4px;
      height: 40px;
      border-radius: 4px;
      background: var(--el-color-primary);
    }
  }

  .image-container:hover .resize-handle {
    opacity: 1;
  }

  .resize-handle--left {
    left: -3px;

    &::after {
      left: 0;
    }
  }

  .resize-handle--right {
    right: -3px;

    &::after {
      right: 0;
    }
  }

  /* Callout blocks */
  .callout-block {
    margin: 12px 0;
    border-radius: 8px;
    border-left: 4px solid var(--art-gray-300);
    background: var(--art-gray-50);
    padding: 0;
  }

  .callout-block__header {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px 0;
    font-size: 13px;
    font-weight: 600;
    user-select: none;
  }

  .callout-block__body {
    padding: 4px 12px 8px;
  }

  .callout-block--info { border-left-color: #3b82f6; background: #eff6ff; }
  .callout-block--tip { border-left-color: #22c55e; background: #f0fdf4; }
  .callout-block--warning { border-left-color: #f59e0b; background: #fffbeb; }
  .callout-block--danger { border-left-color: #ef4444; background: #fef2f2; }
  .callout-block--details { border-left-color: #8b5cf6; background: #f5f3ff; }

  .callout-block--info .callout-block__header { color: #2563eb; }
  .callout-block--tip .callout-block__header { color: #16a34a; }
  .callout-block--warning .callout-block__header { color: #d97706; }
  .callout-block--danger .callout-block__header { color: #dc2626; }
  .callout-block--details .callout-block__header { color: #7c3aed; }
</style>
