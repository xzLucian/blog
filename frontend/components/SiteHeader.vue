<template>
  <header class="site-header">
    <div class="site-header__inner">
      <NuxtLink to="/" class="site-header__brand">
        <img
          :src="isDark ? '/dark-logo.png' : '/light-logo.png'"
          alt="logo"
          class="site-header__logo"
          draggable="false"
        />
      </NuxtLink>

      <nav class="site-header__nav">
        <NuxtLink class="site-header__link" to="/posts">Blog</NuxtLink>
        <NuxtLink class="site-header__link" to="/links">Links</NuxtLink>
        <NuxtLink class="site-header__link" to="/photos">Photos</NuxtLink>
        <NuxtLink class="site-header__link" to="/notes">Notes</NuxtLink>

        <div class="site-header__actions">
          <a
            class="icon-btn"
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            aria-label="GitHub"
            title="GitHub"
          >
            <span class="theme-icon theme-icon--github" aria-hidden="true" />
          </a>
          <button
            type="button"
            class="icon-btn"
            :aria-label="isDark ? 'Switch to light' : 'Switch to dark'"
            @click="toggle($event)"
          >
            <span v-if="isDark" class="theme-icon theme-icon--moon" aria-hidden="true" />
            <span v-else class="theme-icon theme-icon--sun" aria-hidden="true" />
          </button>
        </div>
      </nav>
    </div>
  </header>
</template>

<script setup lang="ts">
const nuxtApp = useNuxtApp()
const isDark = computed(() => nuxtApp.$colorMode?.isDark?.value ?? false)

const toggle = async (event: MouseEvent) => {
  const doToggle = () => nuxtApp.$colorMode?.toggle?.()
  if (!process.client) return
  if (!doToggle) return

  const doc = document
  const root = doc.documentElement
  const x = event.clientX
  const y = event.clientY
  const r = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y))

  root.style.setProperty('--vt-x', `${x}px`)
  root.style.setProperty('--vt-y', `${y}px`)
  root.style.setProperty('--vt-r', `${r}px`)

  const start = (doc as Document & { startViewTransition?: (cb: () => void) => any }).startViewTransition
  if (!start) {
    doToggle()
    return
  }

  const transition = start.call(doc, () => {
    doToggle()
  })

  try {
    await transition?.finished
  } finally {
    root.style.removeProperty('--vt-x')
    root.style.removeProperty('--vt-y')
    root.style.removeProperty('--vt-r')
  }
}
</script>

<style scoped lang="scss">
.site-header {
  position: fixed;
  inset: 0 auto auto 0;
  right: 0;
  top: 0;
  z-index: 50;
  transition:
    transform 220ms ease,
    opacity 220ms ease;
}

.site-header.site-header--hidden {
  transform: translateY(-100%);
  opacity: 0;
  pointer-events: none;
}

.site-header__inner {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.5rem 1.5rem;
}

.site-header__brand {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  user-select: none;
}

.site-header__logo {
  height: 1.75rem;
  width: auto;
}

.site-header__nav {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  font-size: 0.95rem;
  color: rgb(var(--c-muted));
}

.site-header__link {
  color: inherit;
  text-decoration: none;
  transition: color 150ms ease;

  &:hover {
    color: rgb(var(--c-text));
  }
}

.site-header__muted {
  color: rgb(var(--c-muted) / 0.6);
  display: none;
}

.site-header__actions {
  margin-left: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.icon-btn {
  border: none;
  background: transparent;
  padding: 0.5rem;
  border-radius: 0.5rem;
  color: inherit;
  cursor: pointer;
  transition:
    background 150ms ease,
    color 150ms ease;

  &:hover {
    background: rgb(var(--c-card));
    color: rgb(var(--c-text));
  }
}

.icon {
  width: 1rem;
  height: 1rem;
  display: block;
}

.theme-icon {
  --un-icon: none;
  display: inline-block;
  width: 1.2em;
  height: 1.2em;
  vertical-align: text-bottom;
  background-color: currentColor;
  -webkit-mask: var(--un-icon) no-repeat;
  mask: var(--un-icon) no-repeat;
  -webkit-mask-size: 100% 100%;
  mask-size: 100% 100%;
}

.theme-icon--sun {
  --un-icon: url("data:image/svg+xml;utf8,%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20viewBox%3D%270%200%2024%2024%27%3E%3Cpath%20fill%3D%27currentColor%27%20d%3D%27M12%2018a6%206%200%201%201%200-12%206%206%200%200%201%200%2012Zm0-16h0a1%201%200%200%201%201%201v1a1%201%200%201%201-2%200V3a1%201%200%200%201%201-1Zm0%2019a1%201%200%200%201%201%201v1a1%201%200%201%201-2%200v-1a1%201%200%200%201%201-1Zm10-9a1%201%200%200%201-1%201h-1a1%201%200%201%201%200-2h1a1%201%200%200%201%201%201ZM5%2012a1%201%200%200%201-1%201H3a1%201%200%201%201%200-2h1a1%201%200%200%201%201%201Zm14.07-7.07a1%201%200%200%201%200%201.41l-.7.7a1%201%200%201%201-1.42-1.41l.7-.7a1%201%200%200%201%201.42%200ZM7.05%2017.95a1%201%200%200%201%200%201.42l-.7.7a1%201%200%200%201-1.41-1.42l.7-.7a1%201%200%200%201%201.41%200ZM19.07%2019.37a1%201%200%200%201-1.41%200l-.7-.7a1%201%200%201%201%201.41-1.42l.7.7a1%201%200%200%201%200%201.42ZM7.05%206.05a1%201%200%200%201-1.41%200l-.7-.7A1%201%200%201%201%206.35%203.94l.7.7a1%201%200%200%201%200%201.41Z%27/%3E%3C/svg%3E");
}

.theme-icon--moon {
  --un-icon: url("data:image/svg+xml;utf8,%3Csvg viewBox='0 0 24 24' display='inline-block' height='1.2em' width='1.2em' vertical-align='text-bottom' xmlns='http://www.w3.org/2000/svg' %3E%3Cpath fill='currentColor' d='M10 7a7 7 0 0 0 12 4.9v.1c0 5.523-4.477 10-10 10S2 17.523 2 12S6.477 2 12 2h.1A6.98 6.98 0 0 0 10 7m-6 5a8 8 0 0 0 15.062 3.762A9 9 0 0 1 8.238 4.938A8 8 0 0 0 4 12'/%3E%3C/svg%3E");
}

.theme-icon--github {
  --un-icon: url("data:image/svg+xml;utf8,%3Csvg viewBox='0 0 24 24' display='inline-block' height='1.2em' width='1.2em' vertical-align='text-bottom' xmlns='http://www.w3.org/2000/svg' %3E%3Cpath fill='currentColor' d='M10.07 20.503a1 1 0 0 0-1.18-.983c-1.31.24-2.963.276-3.402-.958a5.7 5.7 0 0 0-1.837-2.415a1 1 0 0 1-.167-.11a1 1 0 0 0-.93-.645h-.005a1 1 0 0 0-1 .995c-.004.815.81 1.338 1.141 1.514a4.4 4.4 0 0 1 .924 1.36c.365 1.023 1.423 2.576 4.466 2.376l.003.098l.004.268a1 1 0 0 0 2 0l-.005-.318c-.005-.19-.012-.464-.012-1.182M20.737 5.377q.049-.187.09-.42a6.3 6.3 0 0 0-.408-3.293a1 1 0 0 0-.615-.58c-.356-.12-1.67-.357-4.184 1.25a13.9 13.9 0 0 0-6.354 0C6.762.75 5.455.966 5.102 1.079a1 1 0 0 0-.631.584a6.3 6.3 0 0 0-.404 3.357q.037.191.079.354a6.27 6.27 0 0 0-1.256 3.83a8 8 0 0 0 .043.921c.334 4.603 3.334 5.984 5.424 6.459a5 5 0 0 0-.118.4a1 1 0 0 0 1.942.479a1.7 1.7 0 0 1 .468-.878a1 1 0 0 0-.546-1.745c-3.454-.395-4.954-1.802-5.18-4.899a7 7 0 0 1-.033-.738a4.26 4.26 0 0 1 .92-2.713a3 3 0 0 1 .195-.231a1 1 0 0 0 .188-1.025a3.4 3.4 0 0 1-.155-.555a4.1 4.1 0 0 1 .079-1.616a7.5 7 0 0 1 2.415 1.18a1 1 0 0 0 .827.133a11.8 11.8 0 0 1 6.173.001a1 1 0 0 0 .83-.138a7.6 7.6 0 0 1 2.406-1.19a4 4 0 0 1 .087 1.578a3.2 3.2 0 0 1-.169.607a1 1 0 0 0 .188 1.025c.078.087.155.18.224.268A4.12 4.12 0 0 1 20 9.203a7 7 0 0 1-.038.777c-.22 3.056-1.725 4.464-5.195 4.86a1 1 0 0 0-.546 1.746a1.63 1.63 0 0 1 .466.908a3 3 0 0 1 .093.82v2.333c-.01.648-.01 1.133-.01 1.356a1 1 0 1 0 2 0c0-.217 0-.692.01-1.34v-2.35a5 5 0 0 0-.155-1.311a4 4 0 0 0-.116-.416a6.51 6.51 0 0 0 5.445-6.424A9 9 0 0 0 22 9.203a6.13 6.13 0 0 0-1.263-3.826'/%3E%3C/svg%3E");
}

@media (min-width: 640px) {
  .site-header__muted {
    display: inline;
  }
}
</style>
