<!-- 系统logo -->
<template>
  <div class="flex-cc art-logo-wrap">
    <img :style="{ height: `${size}px` }" :src="menuOpen ? (isDark ? darkLogo : lightLogo) : webTitle" alt="logo" />
  </div>
</template>

<script setup lang="ts">
  import { useSettingStore } from '@/store/modules/setting'
  import lightLogo from '@imgs/common/light-logo.png'
  import darkLogo from '@imgs/common/dark-logo.png'
  import webTitle from '@imgs/common/web-title.png'

  defineOptions({ name: 'ArtLogo' })

  interface Props {
    size?: number | string
  }

  const props = withDefaults(defineProps<Props>(), {
    size: 28
  })

  const settingStore = useSettingStore()
  const { isDark, menuOpen } = storeToRefs(settingStore)
</script>

<style scoped>
.art-logo-wrap img {
  object-fit: contain;
  transition: opacity 0.3s ease;
}
</style>
