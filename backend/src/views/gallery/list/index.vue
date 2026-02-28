<template>
  <div class="page-content !mb-5">
    <div class="gallery-header">
      <span class="gallery-title">图片列表</span>
      <div class="header-actions">
      <button class="view-toggle" @click="viewMode = viewMode === 'uniform' ? 'waterfall' : 'uniform'">
        <svg v-if="viewMode === 'uniform'" viewBox="0 0 24 24" width="1.2em" height="1.2em" xmlns="http://www.w3.org/2000/svg">
          <path fill="currentColor" d="M22 20a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h18a1 1 0 0 1 1 1zm-11-5H4v4h7zm9-4h-7v8h7zm-9-6H4v8h7zm9 0h-7v4h7z" />
        </svg>
        <svg v-else viewBox="0 0 24 24" width="1.2em" height="1.2em" xmlns="http://www.w3.org/2000/svg">
          <path fill="currentColor" d="M14 10h-4v4h4zm2 0v4h3v-4zm-2 9v-3h-4v3zm2 0h3v-3h-3zM14 5h-4v3h4zm2 0v3h3V5zm-8 5H5v4h3zm0 9v-3H5v3zM8 5H5v3h3zM4 3h16a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1" />
        </svg>
      </button>
      <button class="view-toggle" @click="fileInput?.click()">
        <svg viewBox="0 0 24 24" width="1.2em" height="1.2em" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path fill-rule="evenodd" clip-rule="evenodd" d="M20 2C20 1.44772 19.5523 1 19 1C18.4477 1 18 1.44772 18 2V4H16C15.4477 4 15 4.44772 15 5C15 5.55228 15.4477 6 16 6H18V8C18 8.55228 18.4477 9 19 9C19.5523 9 20 8.55228 20 8V6H22C22.5523 6 23 5.55228 23 5C23 4.44772 22.5523 4 22 4H20V2ZM5 4C4.73478 4 4.48043 4.10536 4.29289 4.29289C4.10536 4.48043 4 4.73478 4 5V19C4 19.2652 4.10536 19.5196 4.29289 19.7071C4.48043 19.8946 4.73478 20 5 20H5.58579L14.379 11.2068C14.9416 10.6444 15.7045 10.3284 16.5 10.3284C17.2955 10.3284 18.0584 10.6444 18.621 11.2068L20 12.5858V12C20 11.4477 20.4477 11 21 11C21.5523 11 22 11.4477 22 12V14.998C22 14.9994 22 15.0007 22 15.002V19C22 19.7957 21.6839 20.5587 21.1213 21.1213C20.5587 21.6839 19.7957 22 19 22H6.00219C6.00073 22 5.99927 22 5.99781 22H5C4.20435 22 3.44129 21.6839 2.87868 21.1213C2.31607 20.5587 2 19.7957 2 19V5C2 4.20435 2.31607 3.44129 2.87868 2.87868C3.44129 2.31607 4.20435 2 5 2H12C12.5523 2 13 2.44772 13 3C13 3.55228 12.5523 4 12 4H5ZM8.41422 20H19C19.2652 20 19.5196 19.8946 19.7071 19.7071C19.8946 19.5196 20 19.2652 20 19V15.4142L17.207 12.6212C17.0195 12.4338 16.7651 12.3284 16.5 12.3284C16.2349 12.3284 15.9806 12.4337 15.7931 12.6211L8.41422 20ZM6.87868 6.87868C7.44129 6.31607 8.20435 6 9 6C9.79565 6 10.5587 6.31607 11.1213 6.87868C11.6839 7.44129 12 8.20435 12 9C12 9.79565 11.6839 10.5587 11.1213 11.1213C10.5587 11.6839 9.79565 12 9 12C8.20435 12 7.44129 11.6839 6.87868 11.1213C6.31607 10.5587 6 9.79565 6 9C6 8.20435 6.31607 7.44129 6.87868 6.87868ZM9 8C8.73478 8 8.48043 8.10536 8.29289 8.29289C8.10536 8.48043 8 8.73478 8 9C8 9.26522 8.10536 9.51957 8.29289 9.70711C8.48043 9.89464 8.73478 10 9 10C9.26522 10 9.51957 9.89464 9.70711 9.70711C9.89464 9.51957 10 9.26522 10 9C10 8.73478 9.89464 8.48043 9.70711 8.29289C9.51957 8.10536 9.26522 8 9 8Z"/></svg>
      </button>
      </div>
    </div>

    <input ref="fileInput" type="file" accept="image/*" multiple hidden @change="onFileChange" />

    <VueDraggable v-model="images" :class="['gallery-grid', viewMode]" :animation="200" filter=".upload-card" @end="onDragEnd">
      <div v-for="item in images" :key="item.id" class="grid-card">
        <img :src="proxyUrl + item.url" :alt="item.name" />
        <div class="card-overlay">
          <ElButton type="danger" size="small" circle @click.stop="deleteImage(item)">
            <el-icon><Delete /></el-icon>
          </ElButton>
        </div>
      </div>
    </VueDraggable>
  </div>
</template>

<script setup lang="ts">
  import { ElMessage, ElMessageBox } from 'element-plus'
  import { Delete } from '@element-plus/icons-vue'
  import { VueDraggable } from 'vue-draggable-plus'
  import request from '@/utils/http'

  interface Image {
    id: number
    url: string
    name: string
    sort_order: number
    created_at: string
  }

  const proxyUrl = import.meta.env.VITE_API_PROXY_URL || ''
  const viewMode = ref<'uniform' | 'waterfall'>('uniform')
  const images = ref<Image[]>([])
  const fileInput = ref<HTMLInputElement>()

  const loadImages = async () => {
    const res = await request.get<Image[]>({ url: '/api/images' })
    images.value = res
  }

  const onFileChange = async (e: Event) => {
    const files = (e.target as HTMLInputElement).files
    if (!files?.length) return
    for (const file of files) {
      const formData = new FormData()
      formData.append('file', file)
      const res = await request.post<Image>({ url: '/api/images', data: formData, headers: { 'Content-Type': 'multipart/form-data' } })
      images.value.push(res)
    }
    fileInput.value!.value = ''
    ElMessage.success('上传成功')
  }

  const onDragEnd = async () => {
    const items = images.value.map((img, i) => ({ id: img.id, sort_order: i + 1 }))
    await request.put({ url: '/api/images/sort', data: { items } })
  }

  const deleteImage = async (item: Image) => {
    await ElMessageBox.confirm('确定删除该图片？', '提示', { type: 'warning' })
    await request.del({ url: `/api/images/${item.id}` })
    images.value = images.value.filter(i => i.id !== item.id)
    ElMessage.success('删除成功')
  }

  onMounted(loadImages)
</script>

<style scoped>
.gallery-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.header-actions {
  display: flex;
  gap: 4px;
}
.gallery-title {
  font-size: 16px;
  font-weight: 600;
}
.view-toggle {
  background: none;
  border: none;
  cursor: pointer;
  padding: 6px;
  border-radius: 50%;
  color: var(--el-text-color-secondary);
  display: flex;
  align-items: center;
  justify-content: center;
}
.view-toggle svg {
  width: 18px;
  height: 18px;
}
.view-toggle:last-child svg {
  width: 16px;
  height: 16px;
  stroke: currentColor;
  stroke-width: 0.4px;
}
.view-toggle:hover {
  color: var(--el-color-primary);
  background: var(--el-fill-color-light);
}
.gallery-grid.uniform {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
}
.uniform .grid-card {
  aspect-ratio: 1;
  overflow: hidden;
}
.uniform .grid-card img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.gallery-grid.waterfall {
  columns: 4;
  column-gap: 12px;
}
.waterfall .grid-card {
  break-inside: avoid;
  margin-bottom: 12px;
}
.waterfall .grid-card img {
  width: 100%;
  display: block;
}
.grid-card {
  position: relative;
  overflow: hidden;
  cursor: grab;
}
.card-overlay {
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity 0.2s;
}
.grid-card:hover .card-overlay {
  opacity: 1;
}
</style>
