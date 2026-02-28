<template>
  <div v-if="enabled && show" class="scroll-tools">
    <button
      type="button"
      class="scroll-bubble"
      aria-label="Back to top"
      @click="scrollToTop"
    >
      <span class="scroll-bubble__icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" class="scroll-bubble__svg">
          <path
            d="M12 4l-6.5 6.5m6.5-6.5l6.5 6.5M12 4v16"
            fill="none"
            stroke="currentColor"
            stroke-width="1.8"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
      </span>
    </button>
  </div>
</template>

<script setup lang="ts">
const props = defineProps<{
  enabled?: boolean
}>()

const enabled = computed(() => props.enabled !== false)

const show = ref(false)

let ticking = false

const onScroll = () => {
  if (!ticking) {
    ticking = true
    window.requestAnimationFrame(() => {
      show.value = (window.scrollY || 0) > 220
      ticking = false
    })
  }
}

const scrollToTop = () => {
  window.scrollTo({ top: 0, behavior: 'smooth' })
}

onMounted(() => {
  show.value = (window.scrollY || 0) > 220
  window.addEventListener('scroll', onScroll, { passive: true })
  window.addEventListener('resize', onScroll, { passive: true })
})

onBeforeUnmount(() => {
  window.removeEventListener('scroll', onScroll)
  window.removeEventListener('resize', onScroll)
})
</script>

<style scoped lang="scss">
.scroll-tools {
  position: fixed;
  right: 1.5rem;
  bottom: 2rem;
  z-index: 40;
}

.scroll-bubble {
  width: 2.85rem;
  height: 2.85rem;
  border-radius: 999px;
  border: none;
  background: transparent;
  color: rgb(var(--c-muted) / 0.9);
  cursor: pointer;
  display: grid;
  place-items: center;
  transition:
    background 150ms ease,
    color 150ms ease;

  &:hover {
    background: rgb(var(--c-text) / 0.06);
    color: rgb(var(--c-text));
  }
}

.scroll-bubble__icon {
  display: grid;
  place-items: center;
}

.scroll-bubble__svg {
  width: 1.35rem;
  height: 1.35rem;
  display: block;
}
</style>
