<template>
  <div class="photos-page">
    <button
      type="button"
      class="photos-mode"
      :aria-label="mode === 'tight' ? 'Switch to loose layout' : 'Switch to tight layout'"
      @click="toggleMode"
    >
      <span
        class="photos-icon"
        :class="mode === 'tight' ? 'photos-icon--tight' : 'photos-icon--loose'"
        aria-hidden="true"
      />
    </button>

    <div class="photos-content">
      <div :class="mode === 'tight' ? 'photos-grid' : 'photos-masonry'">
        <figure v-for="p in photos" :key="p.src" class="photo">
          <img class="photo__img" :src="p.src" :alt="p.alt" loading="lazy" />
        </figure>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
useHead({ title: 'Photos' })
type ImageItem = { id: number; url: string; name: string; sort_order: number }

const mode = ref<'tight' | 'loose'>('tight')
const toggleMode = () => {
  mode.value = mode.value === 'tight' ? 'loose' : 'tight'
}

const { data } = useApi<ImageItem[]>('/api/images')

const photos = computed(() =>
  (data.value ?? []).map(img => ({
    src: img.url.startsWith('http')
      ? img.url
      : img.url.startsWith('/')
        ? img.url
        : `/${img.url}`,
    alt: img.name,
  }))
)
</script>

<style scoped lang="scss">
.photos-page {
  width: 100%;
  max-width: 92rem;
  margin: 0 auto;
  padding: 6.5rem 1.5rem 4rem;
}

.photos-content {
  max-width: 74rem;
  margin: 0 auto;
}

.photos-mode {
  position: fixed;
  left: 1.25rem;
  top: 6.25rem;
  width: 3.75rem;
  height: 3.75rem;
  border-radius: 999px;
  border: none;
  background: transparent;
  color: rgb(var(--c-muted) / 0.75);
  font-size: 1.1rem;
  display: grid;
  place-items: center;
  cursor: pointer;
  z-index: 60;
  transition:
    background 150ms ease,
    color 150ms ease;

  &:hover {
    background: rgb(var(--c-muted) / 0.18);
    color: rgb(var(--c-muted));
  }
}

.photos-icon {
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

.photos-icon--tight {
  --un-icon: url("data:image/svg+xml;utf8,%3Csvg viewBox='0 0 24 24' display='inline-block' height='1.2em' width='1.2em' vertical-align='text-bottom' xmlns='http://www.w3.org/2000/svg' %3E%3Cpath fill='currentColor' d='M14 10h-4v4h4zm2 0v4h3v-4zm-2 9v-3h-4v3zm2 0h3v-3h-3zM14 5h-4v3h4zm2 0v3h3V5zm-8 5H5v4h3zm0 9v-3H5v3zM8 5H5v3h3zM4 3h16a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1'/%3E%3C/svg%3E");
}

.photos-icon--loose {
  --un-icon: url("data:image/svg+xml;utf8,%3Csvg viewBox='0 0 24 24' display='inline-block' height='1.2em' width='1.2em' vertical-align='text-bottom' xmlns='http://www.w3.org/2000/svg' %3E%3Cpath fill='currentColor' d='M22 20a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h18a1 1 0 0 1 1 1zm-11-5H4v4h7zm9-4h-7v8h7zm-9-6H4v8h7zm9 0h-7v4h7z'/%3E%3C/svg%3E");
}

.photos-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1.75rem 2.25rem;
}

.photos-masonry {
  column-count: 4;
  column-gap: 2.25rem;
}

.photo {
  margin: 0;
}

.photos-masonry .photo {
  break-inside: avoid;
  margin-bottom: 2.25rem;
}

.photo__img {
  width: 100%;
  display: block;
  border-radius: 0.15rem;
}

.photos-grid .photo {
  overflow: hidden;
  border-radius: 0.15rem;
  background: rgb(var(--c-card) / 0.6);
}

.photos-grid .photo__img {
  aspect-ratio: 4 / 3;
  height: 100%;
  object-fit: cover;
}

@media (max-width: 1100px) {
  .photos-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .photos-masonry {
    column-count: 3;
  }
}

@media (max-width: 800px) {
  .photos-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1.25rem 1.25rem;
  }

  .photos-masonry {
    column-count: 2;
    column-gap: 1.25rem;
  }

  .photos-masonry .photo {
    margin-bottom: 1.25rem;
  }
}

@media (max-width: 520px) {
  .photos-grid {
    grid-template-columns: 1fr;
  }

  .photos-masonry {
    column-count: 1;
  }
}
</style>
