<template>
    <div class="article-share">
        <button :class="['share-button', { copied }, 'flx-center']" @click="copy(currentUrl)"
            :aria-label="copied ? copiedText : text" aria-live="polite">

            <div v-if="!copied" class="flx-center">
                <span class="share-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
                        aria-hidden="true" role="img" class="iconify iconify iconify--solar" alt="Icon" width="1em"
                        height="1em" viewBox="0 0 24 24" style="vertical-align: -0.125em">
                        <path fill="currentColor" fill-rule="evenodd"
                            d="M13.803 5.333c0-1.84 1.5-3.333 3.348-3.333A3.34 3.34 0 0 1 20.5 5.333c0 1.841-1.5 3.334-3.349 3.334a3.35 3.35 0 0 1-2.384-.994l-4.635 3.156a3.34 3.34 0 0 1-.182 1.917l5.082 3.34a3.35 3.35 0 0 1 2.12-.753a3.34 3.34 0 0 1 3.348 3.334C20.5 20.507 19 22 17.151 22a3.34 3.34 0 0 1-3.348-3.333a3.3 3.3 0 0 1 .289-1.356L9.05 14a3.35 3.35 0 0 1-2.202.821A3.34 3.34 0 0 1 3.5 11.487a3.34 3.34 0 0 1 3.348-3.333c1.064 0 2.01.493 2.623 1.261l4.493-3.059a3.3 3.3 0 0 1-.161-1.023"
                            clip-rule="evenodd"></path>
                    </svg>
                </span>
                {{ text }}
            </div>
            <div v-else="copied" class="flx-center">
                <span class="share-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
                        aria-hidden="true" role="img" class="iconify iconify iconify--mdi" alt="Icon" width="1em"
                        height="1em" viewBox="0 0 24 24" style="vertical-align: -0.125em">
                        <path fill="currentColor"
                            d="M23 10a2 2 0 0 0-2-2h-6.32l.96-4.57c.02-.1.03-.21.03-.32c0-.41-.17-.79-.44-1.06L14.17 1L7.59 7.58C7.22 7.95 7 8.45 7 9v10a2 2 0 0 0 2 2h9c.83 0 1.54-.5 1.84-1.22l3.02-7.05c.09-.23.14-.47.14-.73zM1 21h4V9H1z">
                        </path>
                    </svg>
                </span>
                {{ copiedText }}
            </div>
        </button>
    </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useRoute } from 'vitepress';

// 导入useClipboard composable
import { useClipboard } from '../composables/useClipboard';

const text = '分享文章'
const copiedText = '链接已复制!'
const { copy, copied } = useClipboard();
const route = useRoute();

// 获取当前页面完整URL
const currentUrl = computed(() => {
    if (typeof window !== 'undefined') {
        const query = route.path
        const { origin, pathname, search, hash } = location;
        return `${origin}${pathname}${query ? search : ""}${hash ? location.hash : ""}`;
    }
    return '';
});

</script>

<style scoped>
.article-share {
    margin-bottom: 1rem;
}

.share-button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    width: 100%;
    border-radius: 14px;
    padding: 7px 14px;
    border-radius: 14px;
    background-color: var(--vp-c-bg-soft);
    color: #3451b2;
    cursor: pointer;
    transition: all .5s cubic-bezier(.25, .8, .25, 1);
    font-size: 0.875rem;
    font-weight: 500;
}

.share-button:hover:not(:disabled) {
    background-color: var(--vp-c-brand-soft);
    border-color: var(--vp-c-brand-soft);
    color: var(--vp-c-brand-1);
    transform: translateY(-1px);
}

.share-button:active:not(:disabled) {
    transform: scale(0.9);
}

.flx-center {
    display: flex;
    align-items: center;
    justify-content: center;
}

/* 暗黑模式下 */
.dark {
    .share-button {
        color: #a8b1ff;
        background-color: #161618;
    }
}
</style>