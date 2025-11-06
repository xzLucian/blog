<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'

const scrollProgress = ref(0)
const isVisible = ref(false)

const updateScrollProgress = () => {
  const scrollTop = window.scrollY || document.documentElement.scrollTop
  const scrollHeight = document.documentElement.scrollHeight - window.innerHeight
  const progress = (scrollTop / scrollHeight) * 100
  scrollProgress.value = Math.min(Math.round(progress), 100)
  isVisible.value = scrollTop > 200
}

const scrollToTop = () => {
  window.scrollTo({ top: 0, behavior: 'smooth' })
}

onMounted(() => {
  window.addEventListener('scroll', updateScrollProgress)
})

onUnmounted(() => {
  window.removeEventListener('scroll', updateScrollProgress)
})

// SVG 进度条计算
const radius = 18 // 半径减小
const circumference = 2 * Math.PI * radius
const dashOffset = computed(() => circumference - (scrollProgress.value / 100) * circumference)
</script>

<template>
  <div
    v-show="isVisible"
    class="scroll-to-top"
    @click="scrollToTop"
  >
    <svg class="progress-ring" width="50" height="50">
      <circle
        class="progress-ring__background"
        stroke="#ddd"
        stroke-width="3"
        fill="transparent"
        :r="radius"
        cx="25"
        cy="25"
      />
      <circle
        class="progress-ring__progress"
        stroke="#007BFF"
        stroke-width="3"
        fill="transparent"
        :r="radius"
        cx="25"
        cy="25"
        :stroke-dasharray="circumference"
        :stroke-dashoffset="dashOffset"
        stroke-linecap="round"
      />
    </svg>

    <!-- 中间显示百分比 -->
    <div class="progress-text">{{ scrollProgress }}</div>
  </div>
</template>

<style scoped>
.scroll-to-top {
  position: fixed;
  bottom: 35px;
  right: 35px;
  width: 50px;
  height: 50px;
  border-radius: 50%;
  background: white;
  /* box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15); */
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: transform 0.3s ease, opacity 0.3s ease;
  z-index: 9999;
}

.scroll-to-top:hover {
  transform: scale(1.1);
}

.progress-ring {
  position: absolute;
  top: 0;
  left: 0;
  transform: rotate(-90deg);
}

.progress-ring__background {
  opacity: 0.25;
}

.progress-text {
  position: absolute;
  font-size: 13px;
  font-weight: 600;
  color: #007BFF;
  user-select: none;
}
</style>
