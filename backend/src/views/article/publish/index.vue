<!-- 文章发布页面 -->
<template>
  <div class="publish-page">
    <!-- 顶部操作栏 -->
    <div class="publish-header">
      <button class="back-btn" @click="router.back()">
        <ElIcon><ArrowLeft /></ElIcon>
      </button>
      <div class="header-actions">
        <ElButton size="small" @click="saveDraft">
          保存
          <span class="shortcut">⌘+S</span>
        </ElButton>
      </div>
    </div>

    <!-- 编辑器 -->
    <TiptapEditor ref="editorRef" v-model="editorHtml" height="calc(100vh - 320px)" placeholder="开始输入内容...">
      <template #before-content>
        <div class="title-area">
          <input
            v-model="articleName"
            class="title-input"
            placeholder="无标题"
            maxlength="100"
          />
          <div class="meta-row">
            <span class="meta-date">{{ currentDate }}</span>
            <span v-if="showStatus" class="sync-status" :class="{ saved: isSaved, unsaved: !isSaved }">
              <svg v-if="isSaved" xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m17 15-5.5 5.5L9 18" /><path d="M5 17.743A7 7 0 1 1 15.71 10h1.79a4.5 4.5 0 0 1 1.5 8.742" /></svg>
              <svg v-else xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m2 2 20 20" /><path d="M5.782 5.782A7 7 0 0 0 9 19h8.5a4.5 4.5 0 0 0 1.307-.193" /><path d="M21.532 16.5A4.5 4.5 0 0 0 17.5 10h-1.79A7.008 7.008 0 0 0 10 5.07" /></svg>
              {{ isSaved ? '已保存' : `最近更新: ${lastSavedAgo}` }}
            </span>
            <span v-if="currentTypeName" class="meta-tag">{{ currentTypeName }}</span>
            <ElDropdown trigger="click" @command="onTypeCommand">
              <button class="meta-tag-btn">+ 标签</button>
              <template #dropdown>
                <ElDropdownMenu>
                  <ElDropdownItem
                    v-for="item in articleTypes"
                    :key="item.id"
                    :command="item.id"
                  >
                    {{ item.name }}
                  </ElDropdownItem>
                </ElDropdownMenu>
              </template>
            </ElDropdown>
          </div>
        </div>
      </template>
    </TiptapEditor>

    <!-- 底部字数统计 -->
    <div class="publish-footer">
      <span class="word-count">字数: {{ editorRef?.wordCount ?? 0 }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
  import { ArrowLeft } from '@element-plus/icons-vue'
  import { PageModeEnum } from '@/enums/formEnum'
  import request from '@/utils/http'
  import { useCommon } from '@/hooks/core/useCommon'

  defineOptions({ name: 'ArticlePublish' })

  interface ArticleType {
    id: string
    name: string
  }

  const route = useRoute()
  const router = useRouter()

  const editorRef = ref()
  const pageMode = ref<PageModeEnum>(PageModeEnum.Add)
  const articleId = ref<string | undefined>(route.query.id as string | undefined)
  const articleName = ref('')
  const articleType = ref<string>()
  const articleTypes = ref<ArticleType[]>([])
  const editorHtml = ref('')
  const showStatus = ref(false)
  const isSaved = ref(true)
  const lastSavedTime = ref<Date>(new Date())
  const lastSavedAgo = ref('刚刚')

  const currentDate = computed(() => {
    const d = new Date()
    return `${d.getFullYear()}/${String(d.getMonth() + 1).padStart(2, '0')}/${String(d.getDate()).padStart(2, '0')}`
  })

  const currentTypeName = computed(() => articleTypes.value.find((t) => t.id === articleType.value)?.name || '')

  const onTypeCommand = (id: string) => {
    articleType.value = id
  }

  const updateLastSavedAgo = () => {
    const diff = Math.floor((Date.now() - lastSavedTime.value.getTime()) / 1000)
    if (diff < 60) lastSavedAgo.value = '刚刚'
    else if (diff < 3600) lastSavedAgo.value = `${Math.floor(diff / 60)}分钟前`
    else lastSavedAgo.value = `${Math.floor(diff / 3600)}小时前`
  }

  const markSaved = () => {
    showStatus.value = true
    isSaved.value = true
    lastSavedTime.value = new Date()
    lastSavedAgo.value = '刚刚'
  }

  let agoTimer: ReturnType<typeof setInterval>

  watch([articleName, editorHtml, articleType], () => {
    showStatus.value = true
    isSaved.value = false
    updateLastSavedAgo()
  })

  const initPageMode = () => {
    const { id } = route.query
    articleId.value = id as string | undefined
    pageMode.value = id ? PageModeEnum.Edit : PageModeEnum.Add
    if (pageMode.value === PageModeEnum.Edit) getArticleDetail()
  }

  const getArticleTypes = async () => {
    try {
      const res = await request.get<ArticleType[]>({ url: '/api/articles/types' })
      articleTypes.value = res
    } catch {
      ElMessage.error('获取文章分类失败')
    }
  }

  const getArticleDetail = async () => {
    try {
      const res = await request.get<{
        title: string
        blog_class: string
        html_content: string
        status: string
      }>({ url: `/api/articles/${articleId.value}` })
      articleName.value = res.title
      articleType.value = res.blog_class
      editorHtml.value = res.html_content
      nextTick(() => {
        showStatus.value = false
        isSaved.value = true
      })
    } catch {
      ElMessage.error('获取文章详情失败')
    }
  }

  const getArticleData = () => ({
    title: articleName.value,
    blog_class: articleType.value,
    html_content: editorHtml.value,
    type_name: currentTypeName.value
  })

  const saveDraft = async () => {
    if (!articleName.value.trim()) {
      ElMessage.error('请输入文章标题')
      return
    }
    const data = { ...getArticleData(), status: 'draft' }
    try {
      if (articleId.value) {
        await request.put({ url: `/api/articles/${articleId.value}`, data })
      } else {
        const res = await request.post<{ id: number }>({ url: '/api/articles', data })
        articleId.value = String(res.id)
        pageMode.value = PageModeEnum.Edit
      }
      ElMessage.success('草稿保存成功')
      markSaved()
    } catch {
      ElMessage.error('保存失败')
    }
  }

  const { scrollToTop } = useCommon()

  onMounted(() => {
    scrollToTop()
    getArticleTypes()
    initPageMode()
    agoTimer = setInterval(updateLastSavedAgo, 30000)
  })

  onActivated(() => {
    const { id } = route.query
    if (!id) {
      articleId.value = undefined
      articleName.value = ''
      articleType.value = undefined
      editorHtml.value = ''
      showStatus.value = false
      isSaved.value = true
      pageMode.value = PageModeEnum.Add
    } else {
      initPageMode()
    }
  })

  onBeforeUnmount(() => clearInterval(agoTimer))
</script>

<style lang="scss" scoped>
  .publish-page {
    position: fixed;
    inset: 0;
    z-index: 2500;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: var(--default-bg-color);
    font-family: 'DM Sans', sans-serif;
  }

  .publish-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    max-width: 800px;
    margin: 0 auto;
    width: 100%;
    padding: 8px 32px;
    border-bottom: 1px solid var(--art-gray-200);
  }

  .back-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    padding: 0;
    border: none;
    border-radius: 8px;
    background: transparent;
    color: var(--art-gray-700);
    font-size: 18px;
    cursor: pointer;

    &:hover {
      background: var(--art-gray-100);
    }
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 8px;

    .shortcut {
      margin-left: 4px;
      font-size: 11px;
      opacity: 0.6;
    }
  }

  .publish-footer {
    display: flex;
    justify-content: flex-end;
    padding: 6px 24px;
    border-top: 1px solid var(--art-gray-200);
    background: var(--el-bg-color);
  }

  .word-count {
    font-size: 12px;
    color: var(--art-gray-500);
  }
</style>

<style lang="scss">
  .title-area {
    max-width: 800px;
    margin: 0 auto;
    width: 100%;
    padding: 24px 32px 0;
  }

  .title-input {
    width: 100%;
    border: none;
    outline: none;
    font-size: 28px;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    color: var(--art-gray-900);
    background: transparent;

    &::placeholder {
      color: var(--art-gray-400);
    }
  }

  .meta-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 10px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--art-gray-200);
  }

  .meta-date {
    font-size: 13px;
    color: var(--art-gray-400);
  }

  .meta-tag {
    padding: 2px 10px;
    border-radius: 12px;
    background: var(--el-color-primary-light-9);
    color: var(--el-color-primary);
    font-size: 12px;
  }

  .meta-tag-btn {
    padding: 2px 10px;
    border: 1px dashed var(--art-gray-300);
    border-radius: 12px;
    background: transparent;
    color: var(--art-gray-500);
    font-size: 12px;
    cursor: pointer;

    &:hover {
      border-color: var(--el-color-primary);
      color: var(--el-color-primary);
    }
  }

  .sync-status {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;

    &.saved {
      color: var(--el-color-success);
    }

    &.unsaved {
      color: var(--art-gray-400);
    }
  }

  .publish-page .tiptap-toolbar {
    justify-content: center;
  }

  .tiptap-content .tiptap {
    max-width: 800px;
    margin: 0 auto;
    padding: 12px 32px;
  }
</style>
