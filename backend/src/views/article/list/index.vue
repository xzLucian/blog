<!-- 文章列表页面 -->
<template>
  <div class="page-content !mb-5">
    <ElRow justify="space-between" :gutter="10">
      <ElCol :lg="6" :md="6" :sm="14" :xs="16">
        <ElInput
          v-model="searchVal"
          :prefix-icon="Search"
          clearable
          placeholder="输入文章标题查询"
          @keyup.enter="searchArticle"
        />
      </ElCol>
      <ElCol :lg="12" :md="12" :sm="0" :xs="0">
        <div class="custom-segmented">
          <ElSegmented v-model="yearVal" :options="YEAR_OPTIONS" @change="searchArticleByYear" />
        </div>
      </ElCol>
      <ElCol :lg="6" :md="6" :sm="10" :xs="6" style="display: flex; justify-content: end">
        <ElButton @click="toAddArticle">新增文章</ElButton>
      </ElCol>
    </ElRow>

    <div class="mt-5">
      <div
        class="grid grid-cols-5 gap-5 max-2xl:grid-cols-4 max-xl:grid-cols-3 max-lg:grid-cols-2 max-sm:grid-cols-1"
      >
        <div
          class="group c-p overflow-hidden border border-g-300/60 rounded-custom-sm"
          v-for="item in articleList"
          :key="item.id"
          @click="toDetail(item)"
        >
          <div class="card-status-area" :class="item.status === 'published' ? 'is-published' : 'is-draft'">
            <span class="status-badge">{{ item.status === 'published' ? '已发布' : '草稿' }}</span>
            <span v-if="item.type_name" class="type-badge">{{ item.type_name }}</span>
          </div>
          <div class="px-2 py-1">
            <h2 class="text-base text-g-800 font-medium">{{ item.title }}</h2>
            <div class="flex-b w-full h-6 mt-1">
              <div class="flex-c text-g-500">
                <ArtSvgIcon icon="ri:time-line" class="mr-1 text-sm" />
                <span class="text-sm">{{ useDateFormat(item.create_time, 'YYYY-MM-DD') }}</span>
                <div class="w-px h-3 bg-g-400 mx-3.5"></div>
                <ArtSvgIcon icon="ri:eye-line" class="mr-1 text-sm" />
                <span class="text-sm">{{ item.count }}</span>
              </div>
              <ElDropdown trigger="click" @command="(cmd: string) => onCardCommand(cmd, item)">
                <button
                  class="more-btn opacity-0 group-hover:opacity-100"
                  @click.stop
                >···</button>
                <template #dropdown>
                  <ElDropdownMenu>
                    <ElDropdownItem command="publish">发布</ElDropdownItem>
                    <ElDropdownItem command="edit">编辑</ElDropdownItem>
                  </ElDropdownMenu>
                </template>
              </ElDropdown>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div style="margin-top: 16vh" v-if="showEmpty">
      <ElEmpty :description="`未找到相关数据 ${EmojiText[0]}`" />
    </div>

    <div style="display: flex; justify-content: center; margin-top: 20px">
      <ElPagination
        size="default"
        background
        v-model:current-page="currentPage"
        :page-size="pageSize"
        :pager-count="9"
        layout="prev, pager, next, total,jumper"
        :total="total"
        :hide-on-single-page="true"
        @current-change="handleCurrentChange"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
  import { Search } from '@element-plus/icons-vue'
  import { router } from '@/router'
  import { useDateFormat } from '@vueuse/core'
  import EmojiText from '@/utils/ui/emojo'
  import { useCommon } from '@/hooks/core/useCommon'
  import request from '@/utils/http'

  defineOptions({ name: 'ArticleList' })

  interface Article {
    id: number
    home_img: string
    type_name: string
    title: string
    create_time: string
    count: number
    status: string
  }

  interface GetArticleListOptions {
    backTop?: boolean
  }

  const YEAR_OPTIONS = ['All', '2024', '2023', '2022', '2021', '2020', '2019']
  const PAGE_SIZE = 40

  const yearVal = ref('All')
  const searchVal = ref('')
  const articleList = ref<Article[]>([])
  const currentPage = ref(1)
  const pageSize = ref(PAGE_SIZE)
  const total = ref(0)
  const isLoading = ref(true)

  const showEmpty = computed(() => articleList.value.length === 0 && !isLoading.value)

  const getArticleList = async ({ backTop = false }: GetArticleListOptions = {}) => {
    isLoading.value = true

    try {
      if (searchVal.value) {
        yearVal.value = 'All'
      }

      const res = await request.get<{ list: Article[]; total: number }>({
        url: '/api/articles',
        params: {
          page: currentPage.value,
          size: pageSize.value,
          keyword: searchVal.value || undefined,
          year: yearVal.value === 'All' ? undefined : yearVal.value
        }
      })

      articleList.value = res.list
      total.value = res.total

      if (backTop) {
        useCommon().scrollToTop()
      }
    } catch (error) {
      console.error('获取文章列表失败:', error)
    } finally {
      isLoading.value = false
    }
  }

  const searchArticle = () => {
    currentPage.value = 1
    getArticleList({ backTop: true })
  }

  const searchArticleByYear = () => {
    currentPage.value = 1
    getArticleList({ backTop: true })
  }

  const handleCurrentChange = (val: number) => {
    currentPage.value = val
    getArticleList({ backTop: true })
  }

  const toDetail = (item: Article) => {
    router.push({ name: 'ArticleDetail', params: { id: item.id } })
  }

  const toEdit = (item: Article) => {
    router.push({ name: 'ArticlePublish', query: { id: item.id } })
  }

  const publishArticle = async (item: Article) => {
    try {
      await request.put({ url: `/api/articles/${item.id}/publish` })
      ElMessage.success('发布成功')
      getArticleList()
    } catch {
      ElMessage.error('发布失败')
    }
  }

  const onCardCommand = (cmd: string, item: Article) => {
    if (cmd === 'edit') toEdit(item)
    else if (cmd === 'publish') publishArticle(item)
  }

  const toAddArticle = () => {
    router.push({ name: 'ArticlePublish' })
  }

  onMounted(() => {
    getArticleList()
  })

  onActivated(() => {
    getArticleList()
  })
</script>

<style lang="scss">
  .custom-segmented .el-segmented {
    height: 40px;
    padding: 6px;

    --el-border-radius-base: 8px;
  }

  .more-btn {
    padding: 2px 8px;
    border: none;
    border-radius: 6px;
    background: transparent;
    color: var(--art-gray-600);
    font-size: 14px;
    font-weight: bold;
    letter-spacing: 1px;
    cursor: pointer;

    &:hover {
      background: var(--art-gray-200);
    }
  }

  .card-status-area {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px;
    border-bottom: 1px solid var(--art-gray-200);

    &.is-published {
      background: var(--el-color-success-light-9);
    }

    &.is-draft {
      background: var(--el-color-warning-light-9);
    }
  }

  .status-badge {
    font-size: 12px;
    font-weight: 500;

    .is-published & {
      color: var(--el-color-success);
    }

    .is-draft & {
      color: var(--el-color-warning);
    }
  }

  .type-badge {
    font-size: 11px;
    padding: 1px 8px;
    border-radius: 10px;
    background: rgba(0, 0, 0, 0.06);
    color: var(--art-gray-600);
  }
</style>
