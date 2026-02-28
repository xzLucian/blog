import { Node, mergeAttributes } from '@tiptap/core'
import { VueNodeViewRenderer } from '@tiptap/vue-3'
import { Plugin, PluginKey } from '@tiptap/pm/state'
import { Fragment } from '@tiptap/pm/model'
import CodeGroupView from './CodeGroupView.vue'

const OPEN_RE = /^:::\s*code-group\s*$/
const FENCE_RE = /^(```|~~~)([\w-]+)?(?:\s*\[([^\]]+)\])?\s*$/

function buildCodeBlocks(nodes: any[], schema: any) {
  const blocks: any[] = []
  let inFence = false
  let fence = ''
  let lang = ''
  let title: string | null = null
  let buffer: string[] = []

  const flush = () => {
    const text = buffer.join('\n')
    const attrs = {
      language: lang || null,
      title: title || null,
    }
    const content = text ? schema.text(text) : undefined
    blocks.push(schema.nodes.codeBlock.create(attrs, content))
    buffer = []
    fence = ''
    lang = ''
    title = null
  }

  for (const node of nodes) {
    if (node.type?.name === 'codeBlock' && !inFence) {
      blocks.push(node.copy(node.content))
      continue
    }

    const line = node.textContent || ''
    if (!inFence) {
      const m = line.match(FENCE_RE)
      if (m) {
        inFence = true
        fence = m[1]
        lang = m[2] || ''
        title = m[3] || null
        buffer = []
      }
    } else {
      if (line.trim().startsWith(fence)) {
        flush()
        inFence = false
      } else {
        buffer.push(line)
      }
    }
  }

  if (inFence) flush()
  if (!blocks.length) {
    blocks.push(schema.nodes.codeBlock.create())
  }
  return blocks
}

export const CodeGroup = Node.create({
  name: 'codeGroup',
  group: 'block',
  content: 'codeBlock+',
  defining: true,

  parseHTML() {
    return [
      {
        tag: 'div[data-code-group]',
        contentElement: '.code-group__content',
      },
    ]
  },

  renderHTML({ node, HTMLAttributes }) {
    const tabs: any[] = []
    let idx = 0
    node.content.forEach((child) => {
      if (child.type.name !== 'codeBlock') return
      const lang = child.attrs.language || 'text'
      const title = child.attrs.title || lang
      tabs.push([
        'button',
        {
          class: `code-group__tab${idx === 0 ? ' is-active' : ''}`,
          'data-index': idx,
          'data-lang': lang,
          type: 'button',
          tabindex: '-1',
        },
        ['span', { class: 'code-group__tab-lang' }, lang.toUpperCase()],
        ['span', { class: 'code-group__tab-title' }, title],
      ])
      idx += 1
    })

    return [
      'div',
      mergeAttributes(HTMLAttributes, {
        class: 'code-group',
        'data-code-group': 'true',
      }),
      ['div', { class: 'code-group__tabs', contenteditable: 'false' }, ...tabs],
      ['div', { class: 'code-group__content' }, 0],
    ]
  },

  addNodeView() {
    return VueNodeViewRenderer(CodeGroupView)
  },

  addProseMirrorPlugins() {
    const codeGroupType = this.type
    return [
      new Plugin({
        key: new PluginKey('codeGroupInput'),
        props: {
          handleTextInput(view, from, _to, text) {
            const { state } = view
            const $from = state.doc.resolve(from)
            if ($from.depth < 1) return false
            const parent = $from.parent
            if (parent.type.name !== 'paragraph') return false
            if ($from.node(-1).type.name === 'codeGroup') return false

            const lineText = parent.textContent + text
            if (lineText.trim() !== ':::') return false

            const grandParent = $from.node(-1)
            const closingIdx = $from.index(-1)

            let openIdx = -1
            for (let i = closingIdx - 1; i >= 0; i--) {
              const child = grandParent.child(i)
              if (child.type.name === 'paragraph' && OPEN_RE.test(child.textContent)) {
                openIdx = i
                break
              }
            }
            if (openIdx === -1) return false

            const between: any[] = []
            for (let i = openIdx + 1; i < closingIdx; i++) {
              const c = grandParent.child(i)
              between.push(c.copy(c.content))
            }

            // If there is an unclosed fence inside, treat this ::: as fence close, not code-group close.
            let inFence = false
            let fence = ''
            for (const node of between) {
              const line = node.textContent || ''
              if (!inFence) {
                const m = line.match(FENCE_RE)
                if (m) {
                  inFence = true
                  fence = m[1]
                }
              } else if (line.trim().startsWith(fence)) {
                inFence = false
                fence = ''
              }
            }
            if (inFence) return false

            const base = $from.start(-1)
            let startPos = base
            for (let i = 0; i < openIdx; i++) startPos += grandParent.child(i).nodeSize
            let endPos = base
            for (let i = 0; i <= closingIdx; i++) endPos += grandParent.child(i).nodeSize

            const blocks = buildCodeBlocks(between, state.schema)
            const node = codeGroupType.create({}, Fragment.from(blocks))
            const tr = state.tr.replaceWith(startPos, endPos, node)
            view.dispatch(tr)
            return true
          },
        },
      }),
    ]
  },
})
