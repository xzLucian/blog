<template>
  <div class="art-card h-105 p-4 box-border mb-5 max-sm:mb-4">
    <div class="art-card-header mb-2">
      <div class="title">
        <h4>文章阅读排行</h4>
      </div>
    </div>
    <ArtBarChart
      class="box-border p-2"
      barWidth="50%"
      height="calc(100% - 56px)"
      :showAxisLine="false"
      :data="chartData"
      :xAxisData="xAxisLabels"
    />
  </div>
</template>

<script setup lang="ts">
  import request from '@/utils/http'

  const xAxisLabels = ref<string[]>([])
  const chartData = ref<number[]>([])

  onMounted(async () => {
    const res: any = await request.get({ url: '/api/dashboard/monthly' })
    xAxisLabels.value = res.topArticles.map((a: any) => a.title.length > 6 ? a.title.slice(0, 6) + '…' : a.title)
    chartData.value = res.topArticles.map((a: any) => a.count)
  })
</script>
