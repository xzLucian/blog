<template>
  <section class="home-hero">
    <div class="home-hero__left">
      <div class="home-hero__title-block">
        <div class="home-hero__space">SPACE</div>
        <h1 class="home-hero__title">bytesize</h1>
      </div>

      <div class="home-hero__quote">
        <div class="home-hero__quote-bar" />
        <div class="home-hero__quote-text">
          <div>// "Not every journey begins with a map.</div>
          <div>// Some start with a line of code."</div>
        </div>
      </div>
    </div>

    <PaperStack />
  </section>
</template>

<script setup lang="ts">
import PaperStack from '~/components/PaperStack.vue'
useHead({ title: 'index' })
</script>

<style scoped lang="scss">
.home-hero {
  margin: 0 auto;
  padding: 0 1rem;
  max-width: 96rem;
  min-height: calc(100dvh - 10rem);
  display: grid;
  gap: 2.5rem;
  grid-template-columns: 1.1fr 0.9fr;
  align-items: center;
}

.home-hero__left {
  position: relative;
  transform: translateX(1.25rem);
}

.home-hero__title-block {
  position: relative;
  transform: translate(1.5rem, -0.75rem);
}

.home-hero__space {
  position: absolute;
  left: 0;
  top: -0.4rem;
  z-index: 0;
  font-size: 7.4rem;
  opacity: 0.75;
  user-select: none;
  pointer-events: none;
  font-family: 'Archivo Black', sans-serif;
  color: transparent;
  -webkit-text-stroke: 2.5px rgb(var(--c-text) / 0.92);
  text-stroke: 2.5px rgb(var(--c-text) / 0.92);
  letter-spacing: 0.08em;
}

.home-hero__title {
  position: relative;
  z-index: 1;
  display: inline-block;
  align-top: top;
  margin-left: -0.5rem;
  margin-top: -2.4rem;
  font-size: 6.2rem;
  line-height: 0.72;
  color: rgba(0, 0, 0, 0.95);
  font-family: 'Playfair Display', serif;
  font-style: italic;
  font-weight: 800;
  letter-spacing: -0.03em;
}

.home-hero__quote {
  margin-top: 4.2rem;
  margin-left: 0;
  display: flex;
  gap: 1.25rem;
  max-width: 32rem;
}

.home-hero__quote-bar {
  margin-top: 0.25rem;
  height: 3.5rem;
  width: 0.25rem;
  border-radius: 999px;
  background: rgba(168, 85, 247, 0.7);
  flex-shrink: 0;
}

.home-hero__quote-text {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.95rem;
  line-height: 1.75rem;
  color: rgb(var(--c-muted));
}

@media (min-width: 768px) {
  .home-hero__title-block {
    transform: translate(-0.25rem, -1.5rem);
  }

  .home-hero__space {
    top: 0.2rem;
    font-size: 10.5rem;
  }

  .home-hero__title {
    margin-left: -1.5rem;
    margin-top: -3.6rem;
    font-size: 9rem;
  }

  .home-hero__quote {
    margin-left: -1.5rem;
    margin-top: 4.8rem;
  }
}

@media (max-width: 900px) {
  .home-hero {
    grid-template-columns: 1fr;
  }
}

@media (prefers-color-scheme: dark) {
  .home-hero__title {
    color: #fff;
  }
}
</style>
