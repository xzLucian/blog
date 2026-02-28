<template>
  <div class="nav-page" :class="{ 'nav-page--expanded': expanded }">
    <aside
      class="nav-sidebar"
      @mouseenter="hovering = true"
      @mouseleave="hovering = false"
    >
      <div class="nav-panel">
        <button
          type="button"
          class="nav-toggle"
          @click="collapsed = !collapsed"
          aria-label="Toggle categories"
        >
          <svg viewBox="0 0 24 24" class="nav-toggle__icon" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
            <path stroke-linecap="round" d="M4 7h16M4 12h16M4 17h16" />
          </svg>
        </button>

        <div v-show="expanded" class="nav-list">
          <button
            v-for="(c, index) in categories"
            :key="sectionId(index)"
            type="button"
            class="nav-item"
            :class="{ 'nav-item--expanded': expanded, 'nav-item--active': activeId === sectionId(index) }"
            :title="expanded ? undefined : c.title"
            @click="scrollToId(sectionId(index))"
          >
            <span v-if="expanded" class="nav-item__label">{{ c.title }}</span>
          </button>
        </div>
      </div>
    </aside>

    <section class="nav-content">
      <div
        v-for="(c, index) in categories"
        :key="sectionId(index)"
        :id="sectionId(index)"
        class="nav-section"
      >
        <div class="nav-section__watermark watermark-outline">
          {{ c.title }}
        </div>

        <div class="nav-section__inner">
          <div class="nav-grid">
            <a
              v-for="item in c.items"
              :key="item.link"
              class="nav-item-card"
              :href="item.link"
              target="_blank"
              rel="noreferrer"
            >
              <div class="nav-item-card__icon">
                <img
                  :src="getIconSrc(item.icon)"
                  :alt="item.title"
                  class="nav-item-card__img"
                  loading="lazy"
                  @error="markBrokenIcon"
                />
              </div>
              <div class="nav-item-card__body">
                <div class="nav-item-card__title">{{ item.title }}</div>
                <div v-if="item.desc" class="nav-item-card__desc">{{ item.desc }}</div>
              </div>
            </a>
          </div>
        </div>
      </div>
    </section>

  </div>
</template>

<script setup lang="ts">
useHead({ title: 'Links' })

type NavItem = { title: string; link: string; desc?: string; icon?: string }
type NavCategory = { title: string; items: NavItem[] }
type ApiCategory = { id: number; name?: string; value?: string; sort_order?: number; status?: string | number }
type ApiLink = {
  id: number
  category_id?: number | null
  icon?: string
  title: string
  description?: string
  link: string
  info?: string
  status?: string | number
}

const { data: navCategories } = useApi<ApiCategory[]>('/api/nav/categories')
const { data: navLinks } = useApi<{ list: ApiLink[]; total: number }>('/api/nav/links?size=1000')

const categories = computed<NavCategory[]>(() => {
  const rawCategories = navCategories.value ?? []
  const rawLinks = navLinks.value?.list ?? []
  const isEnabled = (status?: string | number) => status === undefined || status === null || String(status) === '1'
  const enabledCategories = rawCategories.filter(c => isEnabled(c.status))
  const enabledLinks = rawLinks.filter(l => isEnabled(l.status))
  const grouped = new Map<number, NavItem[]>()

  for (const link of enabledLinks) {
    const categoryId = link.category_id ?? -1
    if (!grouped.has(categoryId)) grouped.set(categoryId, [])
    grouped.get(categoryId)?.push({
      title: link.title,
      link: link.link,
      desc: link.description || undefined,
      icon: link.icon || undefined,
    })
  }

  return enabledCategories
    .slice()
    .sort((a, b) => (a.sort_order ?? 0) - (b.sort_order ?? 0))
    .map((category) => ({
      title: category.name || category.value || '未命名',
      items: (grouped.get(category.id) || []).slice(),
    }))
    .filter(category => category.items.length > 0)
})
const collapsed = ref(true)
const hovering = ref(false)
const expanded = computed(() => !collapsed.value || hovering.value)
const activeId = ref('')

const sectionId = (index: number) => `section-${index}`

const scrollToId = (id: string) => {
  const el = document.getElementById(id)
  if (!el) return
  activeId.value = id
  el.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

const DEFAULT_ICON = '/favicon.png'
const brokenIcons = reactive<Record<string, true>>({})

const resolveIconSrc = (icon?: string) => {
  if (!icon) return DEFAULT_ICON
  const trimmed = icon.trim()
  if (!trimmed) return DEFAULT_ICON
  if (
    trimmed.startsWith('http://')
    || trimmed.startsWith('https://')
    || trimmed.startsWith('//')
    || trimmed.startsWith('data:')
  ) return trimmed
  return DEFAULT_ICON
}

const getIconSrc = (icon?: string) => {
  const resolved = resolveIconSrc(icon)
  if (brokenIcons[resolved]) return DEFAULT_ICON
  return resolved
}

const markBrokenIcon = (event: Event) => {
  const target = event.target as HTMLImageElement | null
  const src = target?.currentSrc || target?.src
  if (!src) return
  brokenIcons[src] = true
}

let observer: IntersectionObserver | null = null

const setupObserver = () => {
  observer?.disconnect()
  observer = null

  const targets = categories.value
    .map((_, index) => document.getElementById(sectionId(index)))
    .filter((x): x is HTMLElement => Boolean(x))

  if (!targets.length) return
  activeId.value = targets[0].id

  observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter(e => e.isIntersecting)
        .sort((a, b) => (b.intersectionRatio ?? 0) - (a.intersectionRatio ?? 0))[0]
      if (!visible?.target) return
      activeId.value = (visible.target as HTMLElement).id
    },
    { root: null, threshold: [0.15, 0.3, 0.5], rootMargin: '-15% 0px -70% 0px' },
  )

  for (const t of targets) observer.observe(t)
}

onMounted(() => {
  watch(
    categories,
    async () => {
      await nextTick()
      setupObserver()
    },
    { immediate: true },
  )
})

onBeforeUnmount(() => {
  observer?.disconnect()
  observer = null
})
</script>

<style scoped lang="scss">
.nav-page {
  width: 100%;
  position: relative;
  padding-top: 1.25rem;
}

.nav-sidebar {
  position: fixed;
  left: 1.25rem;
  top: 5.25rem;
  height: calc(100dvh - 7rem);
  z-index: 60;
  pointer-events: none;
  font-family: 'Space Grotesk', 'IBM Plex Sans', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
}

.nav-panel {
  pointer-events: auto;
  width: 3.25rem;
  transition: width 200ms ease;
}

.nav-page--expanded .nav-panel {
  width: 17rem;
}

.nav-toggle {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2.75rem;
  height: 2.75rem;
  border-radius: 0.75rem;
  border: none;
  background: transparent;
  color: rgb(var(--c-muted));
  cursor: pointer;
  transition: color 150ms ease;
}

.nav-toggle:hover {
  color: rgb(var(--c-text));
}

.nav-toggle__icon {
  width: 1.25rem;
  height: 1.25rem;
}

/* scroll-to-top is now provided globally by <ScrollToTop /> in the default layout */

.nav-list {
  margin-top: 0.6rem;
  display: grid;
  gap: 0.35rem;
}

.nav-item {
  width: 100%;
  display: flex;
  align-items: center;
  border-radius: 0.75rem;
  border: none;
  padding: 0.2rem 0.25rem;
  font-size: 0.9rem;
  font-weight: 500;
  color: rgb(var(--c-muted) / 0.85);
  background: transparent;
  cursor: pointer;
  transition: color 150ms ease;
}

.nav-item--expanded {
  justify-content: flex-start;
}

.nav-item:not(.nav-item--expanded) {
  justify-content: center;
}

.nav-item:hover {
  color: rgb(var(--c-text));
}

.nav-item--active {
  color: rgb(var(--c-text));
}

.nav-item__label {
  margin-left: 0.2rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 0.85rem;
  line-height: 1.25;
}

.nav-content {
  width: 100%;
  max-width: 65rem;
  margin: 0 auto;
  padding: 0 1.5rem 3.5rem;
}

.nav-section {
  position: relative;
  padding: 5rem 0;
  scroll-margin-top: 7rem;
}

.nav-section__watermark {
  position: absolute;
  left: 2.75rem;
  top: 0.25rem;
  z-index: 0;
  font-size: clamp(4.5rem, 10vw, 8.5rem);
  font-weight: 900;
  letter-spacing: -0.02em;
  opacity: 0.9;
  user-select: none;
  pointer-events: none;
}

.nav-section__inner {
  position: relative;
  z-index: 1;
}

.nav-grid {
  display: grid;
  gap: 2.25rem 3.25rem;
  grid-template-columns: 1fr;
}

.nav-item-card {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  border-radius: 1rem;
  padding: 0.35rem 0.5rem;
  text-decoration: none;
  transition:
    color 150ms ease,
    background 150ms ease;
}

.nav-item-card:hover {
  background: rgb(var(--c-text) / 0.06);
}

.nav-item-card__icon {
  width: 2.25rem;
  height: 2.25rem;
  flex-shrink: 0;
  border-radius: 0.5rem;
  padding: 0.25rem;
  opacity: 0.7;
}

.nav-item-card__img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.nav-item-card__placeholder {
  width: 100%;
  height: 100%;
  border-radius: 0.5rem;
  background: rgb(var(--c-muted) / 0.12);
}

.nav-item-card__title {
  font-weight: 650;
  color: rgb(var(--c-text) / 0.72);
  transition: color 150ms ease;
}

.nav-item-card__body {
  min-width: 0;
  flex: 1 1 auto;
  overflow: hidden;
}

.nav-item-card:hover .nav-item-card__title {
  color: rgb(var(--c-text));
}

.nav-item-card__desc {
  margin-top: 0.25rem;
  font-size: 0.9rem;
  color: rgb(var(--c-muted) / 0.8);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

@media (min-width: 768px) {
  .nav-section__watermark {
    font-size: clamp(5.5rem, 8vw, 9.5rem);
  }

  .nav-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

@media (max-width: 900px) {
  .nav-sidebar {
    left: 0.75rem;
    top: 4.75rem;
  }

  .nav-content {
    padding: 0 1.25rem 3.5rem;
  }
}
</style>
