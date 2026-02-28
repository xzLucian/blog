<template>
  <div class="notes-page">
    <div class="notes-wrap">
      <header class="notes-head">
        <h1 class="notes-title">Notes</h1>
        <p class="notes-subtitle">Small drafts, prompts, and quick ideas.</p>
      </header>

      <div class="notes-shelf">
        <NuxtLink
          v-for="book in books"
          :key="book.key"
          :to="book.href"
          class="note-book"
        >
          <div class="note-book__tab">
            <span class="note-icon note-icon--book" aria-hidden="true" />
            <span class="note-book__tab-title">{{ book.key }}</span>
            <span class="note-book__tab-count">{{ book.count }}</span>
          </div>

          <div class="note-book__body">
            <div class="note-book__preview">{{ book.description }}</div>
          </div>
        </NuxtLink>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
useHead({ title: 'Notes' })
type NoteItem = {
  id: number
  parent_id: number | null
  title: string
  content: string
  sort_order: number
  status: number
}

const { data } = useApi<NoteItem[]>('/api/notes')

const books = computed(() => {
  const items = data.value ?? []
  const parents = items.filter(n => !n.parent_id && n.title)
  const children = items.filter(n => n.parent_id)

  return parents
    .sort((a, b) => a.sort_order - b.sort_order)
    .map(p => {
      const kids = children.filter(c => c.parent_id === p.id)
      return {
        key: p.title,
        href: `/notes/${p.id}`,
        count: `${kids.length} chapters`,
        description: p.content || 'A notebook with multiple chapters.',
      }
    })
})
</script>

<style scoped lang="scss">
.notes-page {
  width: 100%;
  margin: 0 auto;
}

.notes-wrap {
  max-width: 78rem;
  margin: 0 auto;
}

.notes-head {
  max-width: 46rem;
  margin: 0 auto 3rem;
  text-align: center;
}

.notes-title {
  font-size: 3.2rem;
  font-weight: 900;
  letter-spacing: -0.03em;
  margin: 0;
}

.notes-subtitle {
  margin: 0.75rem 0 0;
  font-style: italic;
  font-size: 1.15rem;
  color: rgb(var(--c-muted) / 0.8);
}

.notes-shelf {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1.5rem 1.5rem;
  align-items: start;
}

.note-book {
  position: relative;
  text-decoration: none;
  border-radius: 1rem;
  border: 1px solid rgb(var(--c-border));
  box-shadow: 0 26px 60px rgb(0 0 0 / 0.1);
  overflow: hidden;
  min-height: 10.5rem;
  transform: translateZ(0);
  transition:
    transform 180ms ease,
    box-shadow 180ms ease,
    background 180ms ease;
}

.note-book::before {
  content: '';
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 1.1rem;
  background:
    radial-gradient(circle at 50% 1.1rem, rgb(0 0 0 / 0.14) 0.14rem, transparent 0.16rem),
    radial-gradient(circle at 50% 2.9rem, rgb(0 0 0 / 0.14) 0.14rem, transparent 0.16rem),
    radial-gradient(circle at 50% 4.7rem, rgb(0 0 0 / 0.14) 0.14rem, transparent 0.16rem),
    radial-gradient(circle at 50% 6.5rem, rgb(0 0 0 / 0.14) 0.14rem, transparent 0.16rem),
    radial-gradient(circle at 50% 8.3rem, rgb(0 0 0 / 0.14) 0.14rem, transparent 0.16rem),
    linear-gradient(to right, rgb(0 0 0 / 0.06), transparent);
  opacity: 0.55;
  pointer-events: none;
}

.note-book:hover {
  transform: translateY(-3px);
  box-shadow: 0 34px 80px rgb(0 0 0 / 0.12);
}

.note-book__tab {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  padding: 1rem 1rem 0.6rem 2rem;
  font-family: 'Space Grotesk', 'IBM Plex Sans', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  font-weight: 700;
  color: rgb(var(--c-text));
}

.note-book__tab-title {
  font-size: 1rem;
  letter-spacing: -0.01em;
}

.note-book__tab-count {
  margin-left: auto;
  font-size: 0.9rem;
  font-weight: 650;
  color: rgb(var(--c-muted) / 0.85);
}

.note-book__body {
  padding: 0 1rem 1.2rem 2rem;
}

.note-book__preview {
  color: rgb(var(--c-muted));
  font-size: 0.98rem;
  line-height: 1.55;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.note-icon {
  --un-icon: none;
  display: inline-block;
  width: 1.05rem;
  height: 1.05rem;
  background-color: currentColor;
  -webkit-mask: var(--un-icon) no-repeat;
  mask: var(--un-icon) no-repeat;
  -webkit-mask-size: 100% 100%;
  mask-size: 100% 100%;
  opacity: 0.85;
}

.note-icon--at {
  --un-icon: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='currentColor' d='M12 3a9 9 0 1 0 0 18h3a1 1 0 1 0 0-2h-3a7 7 0 1 1 7-7v1.5a1.5 1.5 0 0 1-3 0V12a5 5 0 1 0-2 4a3.5 3.5 0 0 0 6-2.5V12A9 9 0 0 0 12 3Zm0 6a3 3 0 1 1 0 6a3 3 0 0 1 0-6Z'/%3E%3C/svg%3E");
}

.note-icon--book {
  --un-icon: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='currentColor' d='M6 4h11a2 2 0 0 1 2 2v14a1 1 0 0 1-1.447.894C16.02 20.12 14.46 20 13 20H7a2 2 0 0 0-2 2V6a2 2 0 0 1 1-2Zm0 16.17A3.9 3.9 0 0 1 7 20h6c1.3 0 2.67.09 4 .6V6a1 1 0 0 0-1-1H7a1 1 0 0 0-1 1ZM4 22a1 1 0 0 1-1-1V6a4 4 0 0 1 4-4h10a3 3 0 0 1 3 3v16a1 1 0 0 1-2 0v-.1c-1.33-.54-2.75-.9-5-.9H7a2 2 0 0 0-2 2a1 1 0 0 1-1 1Z'/%3E%3C/svg%3E");
}

.note-icon--write {
  --un-icon: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='currentColor' d='M6 2h9l3 3v15a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2Zm8 1.5V6h2.5L14 3.5ZM7 9h8v2H7V9Zm0 4h8v2H7v-2Z'/%3E%3C/svg%3E");
}

.note-icon--code {
  --un-icon: url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='currentColor' d='M4 6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6Zm4.3 4.3L6.6 12l1.7 1.7l-1.1 1.1L4.4 12l2.8-2.8l1.1 1.1Zm7.4 0l1.1-1.1L19.6 12l-2.8 2.8l-1.1-1.1L17.4 12l-1.7-1.7Z'/%3E%3C/svg%3E");
}

@media (max-width: 980px) {
  .notes-shelf {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 620px) {
  .notes-title {
    font-size: 2.6rem;
  }

  .notes-shelf {
    grid-template-columns: 1fr;
  }
}
</style>
