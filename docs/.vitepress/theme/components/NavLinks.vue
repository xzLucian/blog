<script setup lang="ts">
import { computed } from 'vue'
import { withBase } from 'vitepress'

import type { NavLink } from '../types/NavLink';

// 接收传递过来的参数
const props = defineProps<{
  title: string
  items: NavLink[]
}>()

const formatTitle = computed(() => {
  if (!props.title) {
    return ''
  }
  return props.title
})

</script>

<template>
  <!-- Nav Title -->
  <h2 v-if="title" :id="formatTitle" tabindex="-1">
    {{ title }}
    <a class="header-anchor" :href="`#${formatTitle}`" aria-hidden="true"></a>
  </h2>
  <!-- Nav items -->
  <div class="m-nav-links">
    <template v-for="{ icon, title, desc, link, info } in items">
      <a v-if="link" class="m-nav-link" :href="link" target="_blank" rel="noreferrer">
        <article class="box">
          <div class="box-header">
            <div v-if="icon && typeof icon === 'string'" class="icon">
              <img :src="withBase(icon)" :alt="title" onerror="this.parentElement.style.display='none'" />
            </div>
            <h5 v-if="title" :id="formatTitle" class="title">{{ title }}</h5>
          </div>
          <span v-if="info" class="VPBadge info badge">{{ info }}</span>
          <p v-if="desc" class="desc">{{ desc }}</p>
        </article>
      </a>
    </template>
  </div>
</template>

<style lang="scss" scoped>
.m-nav-links {
  --m-nav-gap: 10px;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
  grid-row-gap: var(--m-nav-gap);
  grid-column-gap: var(--m-nav-gap);
  grid-auto-flow: row dense;
  justify-content: center;
  margin-top: var(--m-nav-gap);
}

@each $media, $size in (500px: 140px, 640px: 155px, 768px: 175px, 960px: 200px, 1440px: 240px) {
  @media (min-width: $media) {
    .m-nav-links {
      grid-template-columns: repeat(auto-fill, minmax($size, 1fr));
    }
  }
}

@media (min-width: 960px) {
  .m-nav-links {
    --m-nav-gap: 20px;
  }
}

.m-nav-link {
  --m-nav-icon-box-size: 40px;
  --m-nav-icon-size: 24px;
  --m-nav-box-gap: 12px;

  display: block;
  border: 1px solid var(--vp-c-bg-soft);
  border-radius: 8px;
  height: 100%;
  text-decoration: inherit;
  background-color: var(--vp-c-bg-alt);
  transition: all 0.25s;

  &:hover {
    box-shadow: var(--vp-shadow-2);
    border-color: var(--vp-c-brand);
    text-decoration: initial;
    background-color: var(--vp-c-bg);
  }

  .box {
    display: flex;
    flex-direction: column;
    padding: var(--m-nav-box-gap);
    height: 100%;
    color: var(--vp-c-text-1);
    position: relative;
    &-header {
      display: flex;
      align-items: center;
    }
  }

  .icon {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-right: calc(var(--m-nav-box-gap) - 2px);
    border-radius: 6px;
    width: var(--m-nav-icon-box-size);
    height: var(--m-nav-icon-box-size);
    font-size: var(--m-nav-icon-size);
    background-color: var(--vp-c-default-soft);
    transition: background-color 0.25s;

    :deep(svg) {
      width: var(--m-nav-icon-size);
      fill: currentColor;
    }

    :deep(img) {
      border-radius: 4px;
      width: var(--m-nav-icon-size);
    }
  }

  .title {
    overflow: hidden;
    flex-grow: 1;
    white-space: nowrap;
    text-overflow: ellipsis;
    line-height: var(--m-nav-icon-box-size);
    font-size: 16px;
    font-weight: 600;
  }

  .desc {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;
    flex-grow: 1;
    margin: calc(var(--m-nav-box-gap) - 2px) 0 0;
    line-height: 1.5;
    font-size: 12px;
    color: var(--vp-c-text-2);
  }
}

@media (max-width: 960px) {
  .m-nav-link {
    --m-nav-icon-box-size: 36px;
    --m-nav-icon-size: 20px;
    --m-nav-box-gap: 8px;

    .title {
      font-size: 14px;
    }
  }
}

.m-nav-link .badge {
    position: absolute;
    top: 2px;
    right: 0;
    transform: scale(.8);
  }
  .VPBadge {
    display: inline-block;
    margin-left: 2px;
    border: 1px solid transparent;
    border-radius: 12px;
    padding: 0 10px;
    line-height: 22px;
    font-size: 12px;
    font-weight: 500;
    transform: translateY(-2px)
}
</style>