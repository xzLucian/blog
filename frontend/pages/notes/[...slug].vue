<template>
  <div class="note-shell">
    <div class="note-layout">
      <aside class="note-left" aria-label="Notebook directory">
        <div class="note-left__inner">
          <div class="note-left__book">
            <div class="note-left__book-title">{{ notebookTitle }}</div>
          </div>

          <nav class="note-left__nav" aria-label="Chapters">
            <div v-if="siblings.length" class="note-left__groups">
              <div class="note-left__group">
                <div class="note-left__group-title">Chapters</div>
                <div class="note-left__items">
                  <NuxtLink
                    v-for="item in siblings"
                    :key="item.id"
                    :to="`/notes/${item.id}`"
                    class="note-left__item"
                    :class="{ 'note-left__item--active': item.id === currentId }"
                  >
                    {{ item.title }}
                  </NuxtLink>
                </div>
              </div>
            </div>
            <div v-else class="note-left__empty">No chapters</div>
          </nav>
        </div>
      </aside>

      <main class="note-center">
        <div class="note-center__inner">
          <div class="note-breadcrumb" aria-label="Breadcrumb">
            <NuxtLink to="/" class="note-breadcrumb__link">Home</NuxtLink>
            <span class="note-breadcrumb__sep" aria-hidden="true">/</span>
            <NuxtLink to="/notes" class="note-breadcrumb__link">Notes</NuxtLink>
            <span class="note-breadcrumb__sep" aria-hidden="true">/</span>
            <span class="note-breadcrumb__current">{{ notebookTitle }}</span>
            <span class="note-breadcrumb__sep" aria-hidden="true">/</span>
            <span class="note-breadcrumb__current">{{ doc?.title ?? 'Note' }}</span>
          </div>

          <article v-if="doc" class="note-article prose note-content">
            <header class="note-head">
              <div class="note-head__row">
                <h1 class="note-title">{{ doc.title }}</h1>
              </div>
            </header>

            <div ref="bodyRef" class="content-body" v-html="doc.content" />
          </article>

          <div v-else class="note-empty">
            <h1 class="note-empty__title">Not found</h1>
            <p class="note-empty__desc">This note does not exist.</p>
          </div>
        </div>
      </main>

      <aside v-if="doc" class="note-right" aria-label="On this page">
        <div class="note-right__inner">
          <button type="button" class="note-right__share" @click="copyLink">
            <span class="note-right__share-icon" aria-hidden="true" />
            <span class="note-right__share-text">{{ copied ? '已复制' : '分享文章' }}</span>
          </button>

          <nav v-if="toc.length" class="note-toc">
            <div class="note-toc__title">On this page</div>
            <div class="note-toc__items">
              <a
                v-for="item in toc"
                :key="item.id"
                :href="`#${item.id}`"
                class="note-toc__item"
                :class="{
                  'note-toc__item--h2': item.level === 2,
                  'note-toc__item--h3': item.level === 3,
                  'note-toc__item--active': item.id === activeId,
                }"
                @click.prevent="scrollTo(item.id)"
              >
                {{ item.text }}
              </a>
            </div>
          </nav>
        </div>
      </aside>
    </div>
  </div>
</template>

<script setup lang="ts">
type NoteItem = {
  id: number
  parent_id: number | null
  title: string
  content: string
  sort_order: number
  status: number
}

const route = useRoute()
const currentId = computed(() => Number(String(route.params.slug ?? '').replace(/,/g, '')))

const { data: allNotes } = useApi<NoteItem[]>('/api/notes')

const doc = computed(() => allNotes.value?.find(n => n.id === currentId.value))

// If landing on a parent note, redirect to its first child
watch([() => currentId.value, allNotes], () => {
  const note = allNotes.value?.find(n => n.id === currentId.value)
  if (!note || note.parent_id) return
  const firstChild = (allNotes.value ?? [])
    .filter(n => n.parent_id === note.id)
    .sort((a, b) => a.sort_order - b.sort_order)[0]
  if (firstChild) navigateTo(`/notes/${firstChild.id}`, { replace: true })
}, { immediate: true })

useHead({ title: computed(() => doc.value?.title ?? '笔记') })

const parentNote = computed(() => {
  if (!doc.value) return null
  if (doc.value.parent_id) {
    return allNotes.value?.find(n => n.id === doc.value!.parent_id) ?? null
  }
  return doc.value
})

const notebookTitle = computed(() => parentNote.value?.title ?? 'Notes')

const siblings = computed(() => {
  const parentId = doc.value?.parent_id ?? doc.value?.id
  if (!parentId) return []
  return (allNotes.value ?? [])
    .filter(n => n.parent_id === parentId)
    .sort((a, b) => a.sort_order - b.sort_order)
})

const copied = ref(false)
const copyLink = async () => {
  if (!import.meta.client) return
  try {
    await navigator.clipboard.writeText(window.location.href)
    copied.value = true
    setTimeout(() => { copied.value = false }, 1200)
  } catch {}
}

const bodyRef = ref<HTMLElement | null>(null)
const { highlight } = useShiki(bodyRef)
watch(doc, () => nextTick(highlight))
onMounted(highlight)

// TOC
type TocItem = { id: string; text: string; level: number }
const toc = ref<TocItem[]>([])
const activeId = ref('')

const buildToc = () => {
  const el = bodyRef.value
  if (!el) { toc.value = []; return }
  const headings = Array.from(el.querySelectorAll<HTMLElement>('h1, h2, h3'))
  toc.value = headings.map((h, i) => {
    if (!h.id) h.id = `heading-${i}`
    const level = h.tagName === 'H1' ? 1 : h.tagName === 'H2' ? 2 : 3
    return { id: h.id, text: h.textContent || '', level }
  })
}

watch(doc, () => nextTick(buildToc))
onMounted(buildToc)

let clickedId = ''

const scrollTo = (id: string) => {
  clickedId = id
  activeId.value = id
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

const onScrollEnd = () => {
  if (!clickedId) return
  activeId.value = clickedId
  clickedId = ''
}

const onScroll = () => {
  if (clickedId) return
  const el = bodyRef.value
  if (!el) return
  const headings = Array.from(el.querySelectorAll<HTMLElement>('h1, h2, h3'))
  if (!headings.length) return
  let current = headings[0].id
  for (const h of headings) {
    if (h.getBoundingClientRect().top <= 120) current = h.id
  }
  activeId.value = current
}

let scrollEndTimer: ReturnType<typeof setTimeout> | null = null
const onScrollWithEnd = () => {
  onScroll()
  if (scrollEndTimer) clearTimeout(scrollEndTimer)
  scrollEndTimer = setTimeout(onScrollEnd, 120)
}

onMounted(() => {
  window.addEventListener('scroll', onScrollWithEnd, { passive: true })
})
onBeforeUnmount(() => {
  window.removeEventListener('scroll', onScrollWithEnd)
  if (scrollEndTimer) clearTimeout(scrollEndTimer)
})
</script>

<style scoped lang="scss">
.note-shell {
  width: 100%;
  margin: 0 auto;
}

.note-layout {
  width: 100%;
  display: grid;
  grid-template-columns: 13.5rem minmax(0, 44rem) 12rem;
  justify-content: center;
  gap: 4rem;
  margin: 0 auto;
}

.note-left {
  width: 12.5rem;
  justify-self: start;
  position: sticky;
  top: 6.5rem;
  align-self: start;
  max-height: calc(100dvh - 8rem);
  overflow: auto;
  padding-right: 0.15rem;
}

.note-right {
  width: 10.75rem;
  justify-self: start;
  position: sticky;
  top: 6.5rem;
  align-self: start;
  max-height: calc(100dvh - 8rem);
  overflow: auto;
  padding-left: 0.15rem;
}

@media (min-width: 981px) {
  .note-left {
    transform: translateX(-1rem);
  }
}

.note-left__inner {
  padding-top: 0.25rem;
}

.note-right__inner {
  padding-top: 0.35rem;
}

.note-left__book {
  margin: 0 0 1.25rem;
}

.note-left__book-title {
  font-size: 1.15rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: rgb(var(--c-text));
}

.note-left__empty {
  color: rgb(var(--c-muted));
  font-size: 0.95rem;
  padding: 0.25rem 0;
}

.note-left__group-title {
  font-size: 0.82rem;
  font-weight: 650;
  color: rgb(var(--c-text) / 0.65);
  margin-bottom: 0.55rem;
  letter-spacing: -0.01em;
}

.note-left__items {
  position: relative;
  display: grid;
  gap: 0.15rem;
  padding: 0.15rem 0 0.15rem 0.6rem;
  border-left: 2px solid rgb(var(--c-border) / 0.9);
}

.note-left__item {
  position: relative;
  text-decoration: none;
  color: rgb(var(--c-muted) / 0.9);
  padding: 0.28rem 0.5rem 0.28rem 0.95rem;
  font-size: 0.94rem;
  font-weight: 550;
  line-height: 1.22rem;
  transition: color 150ms ease;
}

.note-left__item::before {
  content: '';
  position: absolute;
  left: -2px;
  top: 0.4rem;
  bottom: 0.4rem;
  width: 3px;
  border-radius: 999px;
  background: transparent;
}

.note-left__item:hover {
  color: rgb(var(--c-text) / 0.82);
}

.note-left__item--active {
  color: rgb(var(--c-accent-2));
}

.note-left__item--active::before {
  background: rgb(var(--c-accent-2));
}

.note-right__share {
  width: 100%;
  border: none;
  cursor: pointer;
  padding: 0.8rem 0.9rem;
  border-radius: 999px;
  background: rgb(var(--c-text) / 0.04);
  color: rgb(var(--c-accent-2));
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.55rem;
  font-size: 0.86rem;
  font-weight: 750;
  transition: background 150ms ease;
}

.note-right__share:hover {
  background: rgb(var(--c-text) / 0.06);
}

.note-right__share-icon {
  --un-icon: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='currentColor' d='M18 8a3 3 0 1 0-2.83-4H15a3 3 0 0 0 .17 1l-6.6 3.3a3 3 0 1 0 0 7.4l6.6 3.3A3 3 0 1 0 18 16a3 3 0 0 0-2.83 2H15a3 3 0 0 0 .17-1l-6.6-3.3a3 3 0 0 0 0-2.8l6.6-3.3A3 3 0 0 0 18 8Z'/%3E%3C/svg%3E");
  display: inline-block;
  width: 1.2rem;
  height: 1.2rem;
  background-color: currentColor;
  -webkit-mask: var(--un-icon) no-repeat;
  mask: var(--un-icon) no-repeat;
  -webkit-mask-size: 100% 100%;
  mask-size: 100% 100%;
  opacity: 0.9;
}

.note-toc {
  margin-top: 1.5rem;
}

.note-toc__title {
  font-size: 0.8rem;
  font-weight: 700;
  color: rgb(var(--c-text) / 0.85);
  margin-bottom: 0.6rem;
  letter-spacing: -0.01em;
}

.note-toc__items {
  display: grid;
  gap: 0;
  padding-left: 0.6rem;
  border-left: 2px solid rgb(var(--c-border) / 0.9);
}

.note-toc__item {
  position: relative;
  text-decoration: none;
  color: rgb(var(--c-muted) / 0.85);
  padding: 0.25rem 0 0.25rem 0;
  font-size: 0.8rem;
  font-weight: 500;
  line-height: 1.25;
  transition: color 150ms ease;
}

.note-toc__item::before {
  content: '';
  position: absolute;
  left: calc(-0.5rem - 0.6rem - 2px);
  top: 0;
  bottom: 0;
  width: 2px;
  background: transparent;
}

.note-toc__item:hover {
  color: rgb(var(--c-text) / 0.82);
}

.note-toc__item--h2 {
  padding-left: 0.7rem;
}

.note-toc__item--h3 {
  padding-left: 1.4rem;
}

.note-toc__item--active {
  color: rgb(var(--c-accent-2));
}

.note-toc__item--active::before {
  background: rgb(var(--c-accent-2));
}

.note-center {
  min-width: 0;
}

.note-center__inner {
  min-width: 0;
}

.note-breadcrumb {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  font-size: 0.95rem;
  color: rgb(var(--c-muted) / 0.85);
  margin-bottom: 1.65rem;
}

.note-breadcrumb__link {
  text-decoration: none;
  border-bottom: 1px solid transparent;
  transition: border-color 150ms ease;
}

.note-breadcrumb__link:hover {
  border-bottom-color: rgb(var(--c-border));
}

.note-breadcrumb__sep {
  opacity: 0.55;
}

.note-breadcrumb__current {
  color: rgb(var(--c-muted));
}

.note-article {
  max-width: 44rem;
}

.note-head {
  text-align: left;
  margin-bottom: 2.25rem;
}

.note-title {
  font-size: 2.4rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin: 0;
  color: rgb(var(--c-text));
}

.note-empty {
  max-width: 44rem;
  padding: 3rem 0;
  color: rgb(var(--c-muted));
}

.note-empty__title {
  margin: 0;
  font-size: 1.6rem;
  font-weight: 800;
  color: rgb(var(--c-text));
}

.note-empty__desc {
  margin: 0.65rem 0 0;
}

@media (max-width: 980px) {
  .note-layout {
    grid-template-columns: 1fr;
    gap: 2rem;
  }

  .note-left {
    width: 100%;
    justify-self: stretch;
    position: relative;
    top: auto;
    max-height: none;
    overflow: visible;
    padding-right: 0;
  }

  .note-right {
    display: none;
  }
}
</style>
