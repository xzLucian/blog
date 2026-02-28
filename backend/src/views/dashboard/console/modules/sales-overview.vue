<template>
  <div class="art-card h-105 p-5 mb-5 max-sm:mb-4">
    <div class="art-card-header">
      <div class="title">
        <h4>文章发布趋势</h4>
      </div>
    </div>
    <ArtLineChart
      height="calc(100% - 56px)"
      :data="data"
      :xAxisData="xAxisData"
      :showAreaColor="true"
      :showAxisLine="false"
    />
  </div>
</template>

<script setup lang="ts">
  import request from '@/utils/http'

  const xAxisData = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']
  const data = ref<number[]>(new Array(12).fill(0))

  onMounted(async () => {
    const res: any = await request.get({ url: '/api/dashboard/monthly' })
    data.value = res.articles
  })
</script>
