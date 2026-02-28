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

  const initCodeGroups = (root: HTMLElement) => {
    const groups = Array.from(root.querySelectorAll<HTMLElement>('.vp-code-group, .code-group'))
    for (const group of groups) {
      if (group.dataset.codeGroupInit === 'true') continue

      // VitePress-style code groups
      const tabs = Array.from(group.querySelectorAll<HTMLInputElement>('.tabs input[type="radio"]'))
      const blocksWrap = group.querySelector<HTMLElement>('.blocks')
      if (tabs.length && blocksWrap) {
        const blocks = Array.from(blocksWrap.children) as HTMLElement[]
        if (!blocks.length) continue

        const setActive = (index: number) => {
          blocks.forEach((block, i) => block.classList.toggle('active', i === index))
        }

        tabs.forEach((tab, i) => {
          tab.addEventListener('change', () => setActive(i))
        })

        const preset = tabs.findIndex((tab) => tab.checked)
        setActive(preset >= 0 ? preset : 0)
        group.dataset.codeGroupInit = 'true'
        continue
      }

      // code-group: generate tabs from <pre> blocks
      const contentWrap = group.querySelector<HTMLElement>('.code-group__content')
      if (!contentWrap) continue
      const pres = Array.from(contentWrap.querySelectorAll<HTMLElement>(':scope > pre'))
      if (!pres.length) continue

      // Remove any existing tabs div (from old saved HTML)
      group.querySelector('.code-group__tabs')?.remove()

      // Build tabs from pre attributes
      const tabsDiv = document.createElement('div')
      tabsDiv.className = 'code-group__tabs'

      pres.forEach((pre, i) => {
        const lang = pre.getAttribute('data-lang') || 'text'
        const title = pre.getAttribute('data-title') || lang
        const btn = document.createElement('button')
        btn.className = 'code-group__tab' + (i === 0 ? ' is-active' : '')
        btn.type = 'button'
        btn.innerHTML = `<span class="code-group__tab-lang">${lang.toUpperCase()}</span><span class="code-group__tab-title">${title}</span>`
        btn.addEventListener('click', (e) => {
          e.preventDefault()
          setActive(i)
        })
        tabsDiv.appendChild(btn)
      })

      group.insertBefore(tabsDiv, contentWrap)

      const setActive = (index: number) => {
        tabsDiv.querySelectorAll('.code-group__tab').forEach((tab, i) => tab.classList.toggle('is-active', i === index))
        pres.forEach((block, i) => block.classList.toggle('active', i === index))
      }

      setActive(0)
      group.dataset.codeGroupInit = 'true'
    }
  }

  async function highlight() {
    const el = containerRef.value
    if (!el) return
    const blocks = Array.from(el.querySelectorAll('pre:not([data-shiki]) code'))
    for (const code of blocks) {
      const pre = code.parentElement!
      const lang =
        pre.getAttribute('data-lang') ||
        code.getAttribute('data-lang') ||
        code.className.match(/language-([\\w+-]+)/)?.[1] ||
        'text'
      const title = pre.getAttribute('data-title')
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
        if (title) nextPre.setAttribute('data-title', title)
        pre.replaceWith(nextPre)
      } else {
        pre.outerHTML = html
      }
    }

    // enhance generated shiki blocks
    const shikiBlocks = Array.from(el.querySelectorAll('pre.shiki code'))
    for (const code of shikiBlocks) {
      const pre = code.parentElement as HTMLElement
      const lang = pre.getAttribute('data-lang') || code.className.match(/language-(\w+)/)?.[1] || 'text'
      enhance(pre, code as HTMLElement, lang)
    }

    initCodeGroups(el)
  }

  return { highlight }
}
