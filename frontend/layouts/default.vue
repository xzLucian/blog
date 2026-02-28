<template>
  <div class="site-shell">
    <div aria-hidden="true" class="site-bg">
      <div class="site-bg__pattern" :class="bgPatternClass" />
      <div class="site-bg__fade" />
    </div>
    <ClickPop />
    <SiteHeader :class="{ 'site-header--hidden': isHeaderHidden }" />
    <main class="site-main">
      <slot />
    </main>
    <SiteFooter />
    <ScrollToTop :enabled="route.path !== '/'" />
  </div>
</template>

<script setup lang="ts">
import SiteFooter from '~/components/SiteFooter.vue'
import SiteHeader from '~/components/SiteHeader.vue'
import ClickPop from '~/components/ClickPop.vue'
import ScrollToTop from '~/components/ScrollToTop.vue'

const route = useRoute()
const isHeaderHidden = ref(false)

const bgPatternClass = computed(() => {
  if (route.path.startsWith('/posts')) return 'bg-branches'
  if (route.path.startsWith('/links')) return 'bg-dots'
  return 'bg-dots'
})

let lastScroll = 0
let ticking = false

const updateHeaderState = () => {
  const y = window.scrollY || 0

  if (y <= 10) {
    isHeaderHidden.value = false
  } else if (y > lastScroll + 12) {
    isHeaderHidden.value = true
  } else if (y < lastScroll - 12) {
    isHeaderHidden.value = false
  }

  lastScroll = y
}

const onScroll = () => {
  if (!ticking) {
    ticking = true
    window.requestAnimationFrame(() => {
      updateHeaderState()
      ticking = false
    })
  }
}

onMounted(() => {
  lastScroll = window.scrollY || 0
  updateHeaderState()
  window.addEventListener('scroll', onScroll, { passive: true })
  window.addEventListener('resize', onScroll, { passive: true })
})

onBeforeUnmount(() => {
  window.removeEventListener('scroll', onScroll)
  window.removeEventListener('resize', onScroll)
})

watch(
  () => route.path,
  () => {
    isHeaderHidden.value = false
    if (import.meta.client) window.requestAnimationFrame(updateHeaderState)
  }
)
</script>

<style scoped lang="scss">
.site-shell {
  position: relative;
  display: flex;
  flex-direction: column;
  min-height: 100dvh;
  overflow-x: clip;
  background: rgb(var(--c-bg));
  color: rgb(var(--c-text));
}

.site-bg {
  position: fixed;
  inset: 0;
  z-index: -1;
  pointer-events: none;
}

.site-bg__pattern {
  position: absolute;
  inset: 0;
  opacity: 0.8;
}

.site-bg__fade {
  position: absolute;
  inset: 0;
  background: linear-gradient(
    to bottom,
    rgb(var(--c-bg) / 0.3),
    rgb(var(--c-bg) / 0),
    rgb(var(--c-bg))
  );
}

.site-main {
  width: 100%;
  flex: 1 0 auto;
  margin: 0 auto;
  padding: 6rem 1.5rem 0rem 1.5rem;
}
</style>
