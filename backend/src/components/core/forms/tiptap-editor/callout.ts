import { Node, mergeAttributes } from '@tiptap/core'
import { VueNodeViewRenderer } from '@tiptap/vue-3'
import { Plugin, PluginKey } from '@tiptap/pm/state'
import { Fragment } from '@tiptap/pm/model'
import CalloutView from './CalloutView.vue'

export type CalloutType = 'info' | 'tip' | 'warning' | 'danger' | 'details'
const OPEN_RE = /^:::\s*(info|tip|warning|danger|details)\s*(.*)$/

export const Callout = Node.create({
  name: 'callout',
  group: 'block',
  content: 'block+',
  defining: true,

  addAttributes() {
    return {
      type: { default: 'info' },
      title: { default: null },
    }
  },

  parseHTML() {
    return [
      {
        tag: 'div[data-callout]',
        contentElement: '.callout__content',
        getAttrs: (el) => {
          const dom = el as HTMLElement
          return {
            type: dom.getAttribute('data-callout'),
            title: dom.getAttribute('data-title') || dom.querySelector('.callout__title')?.textContent || null,
          }
        },
      },
      {
        tag: 'details.callout',
        contentElement: '.callout__content',
        getAttrs: (el) => {
          const dom = el as HTMLElement
          const summary = dom.querySelector('summary')
          return { type: 'details', title: dom.getAttribute('data-title') || summary?.textContent || null }
        },
      },
    ]
  },

  renderHTML({ node, HTMLAttributes }) {
    const t = node.attrs.type as CalloutType
    const title = node.attrs.title || t.toUpperCase()
    if (t === 'details') {
      return [
        'details',
        mergeAttributes(HTMLAttributes, {
          class: 'callout callout--details',
          'data-callout': 'details',
        }),
        ['summary', { class: 'callout__title' }, title],
        ['div', { class: 'callout__content' }, 0],
      ]
    }
    return [
      'div',
      mergeAttributes(HTMLAttributes, {
        class: `callout callout--${t}`,
        'data-callout': t,
      }),
      ['div', { class: 'callout__title' }, title],
      ['div', { class: 'callout__content' }, 0],
    ]
  },

  addNodeView() {
    return VueNodeViewRenderer(CalloutView)
  },

  addProseMirrorPlugins() {
    const calloutType = this.type
    return [
      new Plugin({
        key: new PluginKey('calloutInput'),
        props: {
          handleTextInput(view, from, _to, text) {
            const { state } = view
            const $from = state.doc.resolve(from)

            if ($from.depth < 1) return false
            const parent = $from.parent
            if (parent.type.name !== 'paragraph') return false
            if ($from.node(-1).type.name === 'callout') return false

            const lineText = parent.textContent + text
            if (lineText.trim() !== ':::') return false

            const grandParent = $from.node(-1)
            const closingIdx = $from.index(-1)

            // Find opening ::: line
            let openIdx = -1
            let openMatch: RegExpMatchArray | null = null
            for (let i = closingIdx - 1; i >= 0; i--) {
              const child = grandParent.child(i)
              if (child.type.name === 'paragraph') {
                const m = child.textContent.match(OPEN_RE)
                if (m) {
                  openIdx = i
                  openMatch = m
                  break
                }
              }
            }
            if (openIdx === -1 || !openMatch) return false

            // Collect content between open and close
            const content: any[] = []
            for (let i = openIdx + 1; i < closingIdx; i++) {
              const c = grandParent.child(i)
              content.push(c.copy(c.content))
            }
            if (!content.length) {
              content.push(state.schema.nodes.paragraph.create())
            }

            // Calculate absolute positions
            const base = $from.start(-1)
            let startPos = base
            for (let i = 0; i < openIdx; i++) {
              startPos += grandParent.child(i).nodeSize
            }
            let endPos = base
            for (let i = 0; i <= closingIdx; i++) {
              endPos += grandParent.child(i).nodeSize
            }

            const node = calloutType.create(
              {
                type: openMatch[1],
                title: openMatch[2]?.trim() || null,
              },
              Fragment.from(content),
            )

            const tr = state.tr.replaceWith(startPos, endPos, node)
            view.dispatch(tr)
            return true
          },
        },
      }),
    ]
  },
})
