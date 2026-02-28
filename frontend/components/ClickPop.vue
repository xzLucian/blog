<template>
  <Teleport to="body">
    <div class="click-pop-layer" aria-hidden="true">
      <div
        v-for="b in bursts"
        :key="b.id"
        class="click-pop"
        :style="{ left: `${b.x}px`, top: `${b.y}px` }"
      >
        <span class="click-pop__ring" />
        <span
          v-for="p in b.particles"
          :key="p.id"
          class="click-pop__p"
          :style="{
            '--dx': `${p.dx}px`,
            '--dy': `${p.dy}px`,
            '--h': `${p.h}deg`,
            '--s': `${p.s}px`,
            '--d': `${p.delay}ms`,
          }"
        />
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
type Particle = { id: string; dx: number; dy: number; h: number; s: number; delay: number }
type Burst = { id: string; x: number; y: number; particles: Particle[] }

const bursts = ref<Burst[]>([])
const reducedMotion = ref(false)

const rand = (min: number, max: number) => min + Math.random() * (max - min)

const createBurst = (x: number, y: number) => {
  const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`
  const count = 10
  const baseHue = Math.floor(rand(0, 360))
  const particles: Particle[] = Array.from({ length: count }).map((_, i) => {
    const a = rand(0, Math.PI * 2)
    const r = rand(18, 40)
    return {
      id: `${id}-${i}`,
      dx: Math.cos(a) * r,
      dy: Math.sin(a) * r,
      h: (baseHue + i * 18) % 360,
      s: rand(3.5, 6.2),
      delay: Math.round(rand(0, 40)),
    }
  })

  bursts.value = [...bursts.value, { id, x, y, particles }].slice(-14)
  window.setTimeout(() => {
    bursts.value = bursts.value.filter(b => b.id !== id)
  }, 650)
}

const shouldIgnoreTarget = (t: EventTarget | null) => {
  const el = t as HTMLElement | null
  if (!el) return false
  return Boolean(el.closest('a, button, input, textarea, select, label, summary'))
}

const onPointerDown = (e: PointerEvent) => {
  if (reducedMotion.value) return
  if (e.button !== 0) return
  if (shouldIgnoreTarget(e.target)) return
  createBurst(e.clientX, e.clientY)
}

onMounted(() => {
  reducedMotion.value = window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches ?? false
  window.addEventListener('pointerdown', onPointerDown, { passive: true })
})

onBeforeUnmount(() => {
  window.removeEventListener('pointerdown', onPointerDown)
})
</script>

<style scoped lang="scss">
.click-pop-layer {
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 20;
}

.click-pop {
  position: fixed;
  width: 1px;
  height: 1px;
}

.click-pop__ring {
  position: absolute;
  inset: 0;
  width: 1px;
  height: 1px;
}

.click-pop__ring::before {
  content: '';
  position: absolute;
  left: 50%;
  top: 50%;
  width: 10px;
  height: 10px;
  border-radius: 999px;
  border: 2px solid rgb(var(--c-text) / 0.12);
  transform: translate(-50%, -50%) scale(0.7);
  opacity: 0.85;
  animation: pop-ring 520ms ease-out forwards;
}

.click-pop__p {
  --dx: 0px;
  --dy: 0px;
  --h: 0deg;
  --s: 5px;
  --d: 0ms;

  position: absolute;
  left: 50%;
  top: 50%;
  width: var(--s);
  height: var(--s);
  border-radius: 999px;
  background: hsl(var(--h) 90% 60%);
  transform: translate(-50%, -50%) translate(0, 0) scale(0.9);
  opacity: 0;
  animation: pop-p 560ms cubic-bezier(0.2, 0.9, 0.2, 1) forwards;
  animation-delay: var(--d);
  filter: drop-shadow(0 10px 18px rgb(0 0 0 / 0.08));
}

@keyframes pop-ring {
  to {
    transform: translate(-50%, -50%) scale(2.8);
    opacity: 0;
  }
}

@keyframes pop-p {
  0% {
    opacity: 0;
    transform: translate(-50%, -50%) translate(0, 0) scale(0.85);
  }
  10% {
    opacity: 1;
  }
  100% {
    opacity: 0;
    transform: translate(-50%, -50%) translate(var(--dx), var(--dy)) scale(0.9);
  }
}

@media (prefers-reduced-motion: reduce) {
  .click-pop-layer {
    display: none;
  }
}
</style>

