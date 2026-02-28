<template>
  <div class="post-page">
    <article class="post-article">
      <header v-if="doc" class="post-header">
        <h1 class="post-title">{{ doc.title }}</h1>
        <div class="post-meta">
          {{ formatDate(doc.create_time) }}
        </div>
      </header>

      <div v-if="doc" ref="bodyRef" class="post-body content-body" v-html="doc.html_content" />
      <div class="post-footer">
        <button type="button" class="post-back" @click="goBack">
          <span class="post-back__icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" class="post-back__svg">
              <path
                d="M8 6l8 6-8 6"
                fill="none"
                stroke="currentColor"
                stroke-width="1.8"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
          </span>
          <span class="post-back__text">cd ..</span>
        </button>
      </div>
    </article>
  </div>
</template>

<script setup lang="ts">
type Article = {
  id: number
  title: string
  html_content: string
  create_time: string
  count: number
  type_name?: string
  status?: string
}

const route = useRoute()
const id = computed(() => String(route.params.slug ?? '').replace(/,/g, ''))

const { data: doc } = useApi<Article>(`/api/articles/${id.value}`)

// 草稿文章不允许前端访问
watch(doc, (val) => {
  if (val && val.status !== 'published') {
    navigateTo('/posts', { replace: true })
  }
}, { immediate: true })

useHead({ title: computed(() => doc.value?.title ?? '文章') })
const router = useRouter()

const formatDate = (iso: string) =>
  new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  }).format(new Date(iso))

const goBack = () => {
  if (process.client && window.history.length > 1) {
    router.back()
  } else {
    router.push('/posts')
  }
}

const bodyRef = ref<HTMLElement | null>(null)
const { highlight } = useShiki(bodyRef)
watch(doc, () => nextTick(highlight))
onMounted(highlight)
</script>

<style scoped lang="scss">
.post-page {
}

.post-footer {
  margin-top: 3.5rem;
  display: flex;
  justify-content: flex-start;
}

.post-back {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  border: none;
  background: transparent;
  color: rgb(var(--c-muted));
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  padding: 0;
}

.post-back__icon {
  display: grid;
  place-items: center;
  color: rgb(var(--c-muted) / 0.7);
}

.post-back__svg {
  width: 1.35rem;
  height: 1.35rem;
  display: block;
}

.post-back__text {
  padding-bottom: 0.15rem;
  border-bottom: 1px solid rgb(var(--c-border));
  letter-spacing: 0.02em;
}

.post-article {
  max-width: 44rem;
  margin: 0 auto;
}

.post-header {
  margin-bottom: 2.75rem;
  text-align: center;
}

.post-title {
  font-size: 2.4rem;
  font-weight: 700;
  letter-spacing: -0.015em;
  color: rgb(var(--c-text));
}

.post-meta {
  margin-top: 0.5rem;
  font-size: 0.95rem;
  color: rgb(var(--c-muted) / 0.75);
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
}

@media (max-width: 600px) {
  .post-back {
    font-size: 0.95rem;
  }

  .post-back__svg {
    width: 1.2rem;
    height: 1.2rem;
  }
}
</style>
