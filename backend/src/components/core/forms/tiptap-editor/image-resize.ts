import { Node, mergeAttributes } from '@tiptap/core'
import { VueNodeViewRenderer } from '@tiptap/vue-3'
import ImageResizeView from './ImageResizeView.vue'

export const ImageResize = Node.create({
  name: 'image',
  group: 'block',
  atom: true,
  draggable: true,

  addAttributes() {
    return {
      src: { default: null },
      alt: { default: null },
      width: { default: null },
      align: { default: 'center' },
    }
  },

  parseHTML() {
    return [{ tag: 'img[src]' }]
  },

  renderHTML({ HTMLAttributes }) {
    const { align, width, ...rest } = HTMLAttributes
    const style = [
      width ? `width: ${width}px` : '',
      align === 'left' ? 'margin-right: auto' : align === 'right' ? 'margin-left: auto' : 'margin: 0 auto',
      'display: block',
    ].filter(Boolean).join('; ')
    return ['img', mergeAttributes(rest, { style })]
  },

  addNodeView() {
    return VueNodeViewRenderer(ImageResizeView)
  },

  addCommands() {
    return {
      setImage: (attrs) => ({ commands }) => {
        return commands.insertContent({ type: this.name, attrs })
      },
    }
  },
})
