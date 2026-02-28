import { ref, type Ref, nextTick, watch, onMounted } from 'vue'
import { codeToHtml } from 'shiki'

export function useShiki(containerRef: Ref<HTMLElement | null>) {
  const enhance = (pre: HTMLElement, codeEl: HTMLElement, lang: string) => {
    if (pre.querySelector('.code-block-toolbar')) return
    pre.classList.add('shiki')
    pre.setAttribute('data-lang', lang || 'text')
    pre.style.position = 'relative'

    const toolbar = document.createElement('div')
    toolbar.className = 'code-block-toolbar'
    toolbar.dataset.lang = lang || 'text'

    const btn = document.createElement('button')
    btn.type = 'button'
    btn.className = 'code-block-copy'
    const label = document.createElement('span')
    label.className = 'code-block-copy-label'
    label.textContent = lang || 'text'
    const icon = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
    icon.setAttribute('viewBox', '0 0 24 24')
    icon.setAttribute('aria-hidden', 'true')
    icon.classList.add('code-block-copy-icon')
    icon.innerHTML = '<path d="M9 9V5a2 2 0 0 1 2-2h7a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-4"/><rect x="4" y="7" width="10" height="12" rx="2" ry="2"/>'
    btn.append(label, icon)

    btn.addEventListener('click', async (e) => {
      e.preventDefault()
      e.stopPropagation()
      try {
        await navigator.clipboard?.writeText(codeEl.innerText || '')
        label.textContent = 'copied'
        toolbar.dataset.copied = 'true'
        setTimeout(() => {
          label.textContent = lang || 'text'
          toolbar.dataset.copied = 'false'
        }, 1500)
      } catch {
        label.textContent = 'fail'
        setTimeout(() => (label.textContent = lang || 'text'), 1200)
      }
    })

    toolbar.append(btn)
    pre.appendChild(toolbar)
  }

  function fixTaskLists(el: HTMLElement) {
    const taskLists = el.querySelectorAll<HTMLElement>('ul[data-type="taskList"]')
    for (const ul of taskLists) {
      ul.style.listStyle = 'none'
      ul.style.paddingLeft = '0'
    }
    const taskItems = el.querySelectorAll<HTMLElement>('li[data-type="taskItem"]')
    for (const li of taskItems) {
      li.style.display = 'flex'
      li.style.flexDirection = 'row'
      li.style.alignItems = 'center'
      li.style.gap = '0.5rem'
      li.style.listStyle = 'none'
      for (const child of Array.from(li.children)) {
        const c = child as HTMLElement
        if (c.tagName === 'LABEL') {
          c.style.display = 'inline-flex'
          c.style.alignItems = 'center'
          c.style.flex = '0 0 auto'
        } else if (c.tagName === 'DIV') {
          c.style.flex = '1 1 0%'
          c.style.minWidth = '0'
          const p = c.querySelector(':scope > p') as HTMLElement | null
          if (p) p.style.margin = '0'
        }
      }
    }
  }

  async function highlight() {
    const el = containerRef.value
    if (!el) return

    // Always fix task lists first, regardless of shiki success
    fixTaskLists(el)

    try {
      const blocks = Array.from(el.querySelectorAll('pre:not([data-shiki]) code'))
      for (const code of blocks) {
        const pre = code.parentElement!
        const lang = code.className.match(/language-(\w+)/)?.[1] || 'text'
        const text = code.textContent || ''
        const html = await codeToHtml(text, {
          lang,
          themes: { light: 'github-light', dark: 'github-dark' },
          defaultColor: false,
        })
        const wrapper = document.createElement('div')
        wrapper.innerHTML = html
        const nextPre = wrapper.firstElementChild as HTMLElement | null
        if (nextPre) {
          nextPre.setAttribute('data-lang', lang)
          pre.replaceWith(nextPre)
        } else {
          pre.outerHTML = html
        }
      }

      const shikiBlocks = Array.from(el.querySelectorAll('pre.shiki code'))
      for (const code of shikiBlocks) {
        const pre = code.parentElement as HTMLElement
        const lang = pre.getAttribute('data-lang') || code.className.match(/language-(\w+)/)?.[1] || 'text'
        enhance(pre, code as HTMLElement, lang)
      }
    } catch (e) {
      console.error('Shiki highlight error:', e)
    }
  }

  return { highlight }
}
