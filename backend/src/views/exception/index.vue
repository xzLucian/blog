<template>
  <div class="flex-cc flex-col h-screen">
    <h1 class="text-6xl font-bold text-gray-300">{{ code }}</h1>
    <p class="mt-4 text-gray-500">{{ message }}</p>
    <ElButton class="mt-6" @click="router.push('/')">返回首页</ElButton>
  </div>
</template>

<script setup lang="ts">
import { router } from '@/router'

const props = defineProps<{ code?: string; message?: string }>()
const code = props.code ?? '404'
const message = props.message ?? '页面不存在'
</script>
