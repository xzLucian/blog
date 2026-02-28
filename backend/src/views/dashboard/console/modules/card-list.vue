<template>
  <ElRow :gutter="20" class="flex">
    <ElCol v-for="(item, index) in dataList" :key="index" :sm="12" :md="6" :lg="6">
      <div class="art-card relative flex flex-col justify-center h-35 px-5 mb-5 max-sm:mb-4">
        <span class="text-g-700 text-sm">{{ item.des }}</span>
        <ArtCountTo class="text-[26px] font-medium mt-2" :target="item.num" :duration="1300" />
        <div
          class="absolute top-0 bottom-0 right-5 m-auto size-12.5 rounded-xl flex-cc bg-theme/10"
        >
          <ArtSvgIcon :icon="item.icon" class="text-xl text-theme" />
        </div>
      </div>
    </ElCol>
  </ElRow>
</template>

<script setup lang="ts">
  import request from '@/utils/http'

  const dataList = reactive([
    { des: '文章数量', icon: 'ri:article-line', num: 0 },
    { des: '笔记数量', icon: 'ri:sticky-note-line', num: 0 },
    { des: '导航数量', icon: 'ri:compass-3-line', num: 0 },
    { des: '图片数量', icon: 'ri:image-line', num: 0 }
  ])

  onMounted(async () => {
    const res: any = await request.get({ url: '/api/dashboard/stats' })
    dataList[0].num = res.articles
    dataList[1].num = res.notes
    dataList[2].num = res.navLinks
    dataList[3].num = res.images
  })
</script>
