<template>
    <nav class="breadcrumb" aria-label="breadcrumb">
        <ol class="breadcrumb-list">
            <li v-for="(item, index) in breadcrumbItems" :key="index" class="breadcrumb-item" :class="{
                'breadcrumb-item--current': item.isCurrent,
                'breadcrumb-item--home': index === 0
            }">
                <a v-if="item.link && !item.isCurrent" :href="item.link" class="breadcrumb-link"
                    @click.prevent="handleClick(item)">
                    {{ item.name }}
                </a>
                <span v-else class="breadcrumb-current" aria-current="page">
                    {{ item.name }}
                </span>
                <span v-if="index < breadcrumbItems.length - 1" class="breadcrumb-separator">
                    /
                </span>
            </li>
        </ol>
    </nav>
</template>

<script setup lang="ts">

import { useRoute, useRouter } from 'vitepress';
import { computed } from 'vue';

const route = useRoute();
const router = useRouter();

// 使用 computed 确保响应式
const breadcrumbItems = computed(() => {
    // console.log('路由路径更新:', route.path); // 调试用

    const path = route.path.replace(/^\/|\/$/g, '');
    const pathArray = path ? path.split('/') : [];

    const items: any = [];
    let accumulatedPath = '';

    // 添加首页项
    items.push({
        name: '首页',
        link: '/',
        isCurrent: false
    });

    // 如果是首页，直接返回
    if (pathArray.length === 0) {
        items[0].isCurrent = true;
        return items;
    }

    // 遍历路径片段
    pathArray.forEach((segment, index) => {
        accumulatedPath += '/' + segment;

        // 美化显示名称
        let displayName = segment;

        // 处理 index 文件的情况
        if (segment === 'index') {
            // 不单独显示 index，而是更新上一个面包屑的链接
            if (items.length > 0) {
                items[items.length - 1].link = accumulatedPath;
            }
            return;
        }

        displayName = displayName.replace(/[-_]/g, ' ');
        displayName = displayName.charAt(0).toUpperCase() + displayName.slice(1);

        const isLastItem = index === pathArray.length - 1;

        items.push({
            name: displayName,
            link: isLastItem ? undefined : accumulatedPath,
            isCurrent: isLastItem
        });
    });

    return items;
});

function handleClick(item: any) {
    if (item.link) {
        router.go(item.link);
    }
}
</script>


<style scoped>
.breadcrumb {
    font-size: 0.875rem;
}

.breadcrumb-list {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.25rem;
}

.breadcrumb-item {
    display: flex;
    align-items: center;
    transition: all 0.2s ease;
}

.breadcrumb-item--home .breadcrumb-link {
    font-weight: 600;
}

.breadcrumb-link {
    color: var(--vp-c-brand-1);
    text-decoration: none;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    transition: all 0.25s;
}

.breadcrumb-link:hover {
    color: var(--vp-c-brand-2);
    background-color: var(--vp-c-bg-mute);
    text-decoration: none;
}

.breadcrumb-current {
    color: var(--vp-c-text-2);
    font-weight: 500;
    padding: 0.25rem 0.5rem;
}

.breadcrumb-separator {
    color: var(--vp-c-text-3);
    margin: 0 0.25rem;
    user-select: none;
}

/* 移动端适配 */

@media (max-width: 768px) {
    .breadcrumb {
        font-size: 0.8rem;
        margin: 0.5rem 0 1rem;
    }

    .breadcrumb-list {
        flex-wrap: wrap;
    }

    .breadcrumb-link,
    .breadcrumb-current {
        padding: 0.125rem 0.25rem;
    }
}
</style>