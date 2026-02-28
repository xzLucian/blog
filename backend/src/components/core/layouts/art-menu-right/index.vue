<template>
  <Teleport to="body">
    <div v-if="visible" class="art-menu-right" :style="menuStyle" @click.stop>
      <div
        v-for="item in menuItems"
        :key="item.key"
        class="menu-item"
        :class="{ disabled: item.disabled, 'show-line': item.showLine }"
        @click="onSelect(item)"
      >
        <ArtSvgIcon v-if="item.icon" :icon="item.icon" class="menu-icon" />
        <span>{{ item.label }}</span>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
defineOptions({ name: 'ArtMenuRight' })

interface MenuItem {
  key: string
  label: string
  icon?: string
  disabled?: boolean
  showLine?: boolean
}

const props = withDefaults(defineProps<{
  menuItems: MenuItem[]
  menuWidth?: number
  borderRadius?: number
}>(), {
  menuWidth: 140,
  borderRadius: 10
})

const emit = defineEmits<{ select: [item: MenuItem] }>()

const visible = ref(false)
const position = ref({ x: 0, y: 0 })

const menuStyle = computed(() => ({
  left: `${position.value.x}px`,
  top: `${position.value.y}px`,
  width: `${props.menuWidth}px`,
  borderRadius: `${props.borderRadius}px`
}))

const show = (e: MouseEvent) => {
  position.value = { x: e.clientX, y: e.clientY }
  visible.value = true
}

const hide = () => { visible.value = false }

const onSelect = (item: MenuItem) => {
  if (item.disabled) return
  emit('select', item)
  hide()
}

const onClickOutside = () => { if (visible.value) hide() }

onMounted(() => document.addEventListener('click', onClickOutside))
onUnmounted(() => document.removeEventListener('click', onClickOutside))

defineExpose({ show, hide })
</script>

<style scoped>
.art-menu-right {
  position: fixed;
  z-index: 9999;
  background: var(--el-bg-color);
  border: 1px solid var(--el-border-color-lighter);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  padding: 6px 0;
}
.menu-item {
  display: flex;
  align-items: center;
  padding: 8px 14px;
  font-size: 13px;
  cursor: pointer;
  color: var(--el-text-color-regular);
  transition: background 0.15s;
}
.menu-item:hover:not(.disabled) {
  background: var(--el-fill-color-light);
  color: var(--el-color-primary);
}
.menu-item.disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.menu-item.show-line {
  border-bottom: 1px solid var(--el-border-color-lighter);
  margin-bottom: 4px;
  padding-bottom: 12px;
}
.menu-icon {
  margin-right: 8px;
  font-size: 15px;
}
</style>
