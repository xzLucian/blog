<!-- 文章详情页面 -->
<template>
  <div class="article-detail page-content">
    <div class="max-w-200 m-auto mt-15">
      <h1 class="text-3xl font-semibold">{{ articleTitle }}</h1>
      <div ref="bodyRef" class="content-body mt-12.5" v-html="articleHtml"></div>
    </div>
    <ArtBackToTop />
  </div>
</template>

<script setup lang="ts">
  import '@/assets/styles/core/content.scss'
  import { useShiki } from '@/hooks/core/useShiki'
  import { useCommon } from '@/hooks/core/useCommon'
  import request from '@/utils/http'

  defineOptions({ name: 'ArticleDetail' })

  const route = useRoute()
  const articleId = computed(() => Number(route.params.id))
  const articleTitle = ref('')
  const articleHtml = shallowRef('')

  const bodyRef = ref<HTMLElement | null>(null)
  const { highlight } = useShiki(bodyRef)

  const getArticleDetail = async () => {
    if (!articleId.value) return

    try {
      const res = await request.get<{ title: string; html_content: string }>({
        url: `/api/articles/${articleId.value}`
      })
      articleTitle.value = res.title
      articleHtml.value = res.html_content
      nextTick(highlight)
    } catch (err) {
      console.error('获取文章详情失败:', err)
    }
  }

  const { scrollToTop } = useCommon()

  onMounted(() => {
    scrollToTop()
    getArticleDetail()
  })
</script>

<style lang="scss">
.content-body ul[data-type="taskList"] {
  list-style: none !important;
  padding-left: 0 !important;
}

.content-body li[data-type="taskItem"] {
  display: flex !important;
  flex-direction: row !important;
  align-items: center !important;
  gap: 0.5rem;
  list-style: none !important;
}

.content-body li[data-type="taskItem"] > label {
  display: inline-flex !important;
  align-items: center;
  flex: 0 0 auto !important;
  cursor: pointer;
}

.content-body li[data-type="taskItem"] > label input[type="checkbox"] {
  width: 1rem;
  height: 1rem;
  margin: 0;
}

.content-body li[data-type="taskItem"] > div {
  flex: 1 1 0% !important;
  min-width: 0;
}

.content-body li[data-type="taskItem"] > div > p {
  margin: 0;
}
</style>
