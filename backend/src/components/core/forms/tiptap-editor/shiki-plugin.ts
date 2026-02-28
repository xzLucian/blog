import { findChildren } from '@tiptap/core'
import { Plugin, PluginKey } from '@tiptap/pm/state'
import { Decoration, DecorationSet } from '@tiptap/pm/view'
import {
  createHighlighterCoreSync,
  createJavaScriptRegexEngine,
  type HighlighterCore,
} from 'shiki'
import js from 'shiki/langs/javascript.mjs'
import ts from 'shiki/langs/typescript.mjs'
import html from 'shiki/langs/html.mjs'
import css from 'shiki/langs/css.mjs'
import json from 'shiki/langs/json.mjs'
import md from 'shiki/langs/markdown.mjs'
import python from 'shiki/langs/python.mjs'
import java from 'shiki/langs/java.mjs'
import go from 'shiki/langs/go.mjs'
import rust from 'shiki/langs/rust.mjs'
import bash from 'shiki/langs/bash.mjs'
import sql from 'shiki/langs/sql.mjs'
import yaml from 'shiki/langs/yaml.mjs'
import vue from 'shiki/langs/vue.mjs'
import jsx from 'shiki/langs/jsx.mjs'
import tsx from 'shiki/langs/tsx.mjs'
import c from 'shiki/langs/c.mjs'
import cpp from 'shiki/langs/cpp.mjs'
import shell from 'shiki/langs/shellscript.mjs'
import githubLight from 'shiki/themes/github-light.mjs'
import githubDark from 'shiki/themes/github-dark.mjs'

const key = new PluginKey('shiki')

let hl: HighlighterCore | null = null

function getHL(): HighlighterCore {
  if (!hl) {
    hl = createHighlighterCoreSync({
      themes: [githubLight, githubDark],
      langs: [js, ts, html, css, json, md, python, java, go, rust, bash, sql, yaml, vue, jsx, tsx, c, cpp, shell],
      engine: createJavaScriptRegexEngine(),
    })
  }
  return hl
}

const SHIKI_THEMES = {
  light: 'github-light',
  dark: 'github-dark',
} as const

function styleObjectToString(style?: Record<string, string | number>) {
  if (!style) return ''
  return Object.entries(style)
    .map(([key, value]) => `${key}: ${String(value)}`)
    .join('; ')
}

function createToolbar(lang: string, text: string) {
  const wrapper = document.createElement('div')
  wrapper.className = 'code-block-toolbar'
  wrapper.dataset.lang = lang || 'text'
  wrapper.setAttribute('contenteditable', 'false')
  wrapper.setAttribute('data-shiki-toolbar', 'true')

  const btn = document.createElement('button')
  btn.className = 'code-block-copy'
  btn.type = 'button'
  btn.tabIndex = -1
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
      await navigator.clipboard?.writeText(text)
      label.textContent = 'copied'
      wrapper.dataset.copied = 'true'
      setTimeout(() => {
        label.textContent = lang || 'text'
        wrapper.dataset.copied = 'false'
      }, 1500)
    } catch {
      label.textContent = 'fail'
      setTimeout(() => (label.textContent = lang || 'text'), 1200)
    }
  })

  wrapper.appendChild(btn)
  return wrapper
}

function getDecorations(doc: any, name: string, highlighter: HighlighterCore) {
  const decorations: Decoration[] = []
  const langs = new Set(highlighter.getLoadedLanguages())

  findChildren(doc, (node) => node.type.name === name).forEach((block) => {
    const lang = block.node.attrs.language
    const text = block.node.textContent
    if (!lang || !langs.has(lang) || !text) return

    const nodeStart = block.pos + 1
    const nodeEnd = nodeStart + text.length

    try {
      const tokens = highlighter.codeToTokens(text, {
        lang,
        themes: SHIKI_THEMES,
        defaultColor: false,
      })
      decorations.push(
        Decoration.widget(nodeStart, () => createToolbar(lang, text), { side: -1, ignoreSelection: true }),
      )
      let from = nodeStart
      const lines = tokens.tokens
      for (let i = 0; i < lines.length; i++) {
        for (const token of lines[i]) {
          const to = from + token.content.length
          const style = styleObjectToString(token.htmlStyle)
          if (style && to <= nodeEnd && from < to) {
            decorations.push(
              Decoration.inline(from, to, { style }),
            )
          }
          from = to
        }
        if (i < lines.length - 1) from += 1
      }
    } catch { /* skip block on error */ }
  })

  return DecorationSet.create(doc, decorations)
}

export function ShikiPlugin(name: string) {
  const plugin: Plugin = new Plugin({
    key,
    state: {
      init(_, state) {
        return getDecorations(state.doc, name, getHL())
      },
      apply(tr, decorationSet) {
        if (tr.docChanged) {
          return getDecorations(tr.doc, name, getHL())
        }
        return decorationSet.map(tr.mapping, tr.doc)
      },
    },
    props: {
      decorations(state) {
        return plugin.getState(state)
      },
    },
  })

  return plugin
}
