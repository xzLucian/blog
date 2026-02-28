<template>
  <div class="page-content !mb-5">
    <ElRow :gutter="16" style="height: calc(100vh - 160px); flex-wrap: nowrap">
      <!-- 左侧：树形目录 -->
      <ElCol :span="5">
        <div class="tree-panel">
          <div class="tree-header">
            <span class="tree-title">笔记目录</span>
            <ElButton size="small" @click="addRootNote">新增</ElButton>
          </div>
          <ElTree
            ref="treeRef"
            :data="treeData"
            node-key="id"
            default-expand-all
            highlight-current
            draggable
            :allow-drag="allowDrag"
            :allow-drop="allowDrop"
            :props="{ label: 'title', children: 'children' }"
            @node-click="onNodeClick"
            @node-drop="onNodeDrop"
            @node-contextmenu="onContextMenu"
          >
            <template #default="{ data }">
              <span class="tree-node-label">{{ data.title || '未命名' }}</span>
            </template>
          </ElTree>
        </div>
      </ElCol>

      <!-- 右侧：编辑区 -->
      <ElCol :span="19">
        <div class="edit-panel" v-if="currentNote">
          <!-- 编辑器 -->
          <TiptapEditor ref="editorRef" v-model="currentNote.content" height="calc(100vh - 350px)" placeholder="开始输入内容...">
            <template #toolbar-extra>
              <div class="toolbar-save">
                <ElButton size="small" @click="saveNote(0)">
                  保存
                  <span class="shortcut">⌘+S</span>
                </ElButton>
              </div>
            </template>
            <template #before-content>
              <div class="title-area">
                <input
                  v-model="currentNote.title"
                  class="title-input"
                  placeholder="无标题"
                />
                <div class="meta-row">
                  <span class="meta-date">{{ formatTime(currentNote.update_time) }}</span>
                  <span v-if="showStatus" class="sync-status" :class="{ saved: isSaved, unsaved: !isSaved }">
                    <svg v-if="isSaved" xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m17 15-5.5 5.5L9 18" /><path d="M5 17.743A7 7 0 1 1 15.71 10h1.79a4.5 4.5 0 0 1 1.5 8.742" /></svg>
                    <svg v-else xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m2 2 20 20" /><path d="M5.782 5.782A7 7 0 0 0 9 19h8.5a4.5 4.5 0 0 0 1.307-.193" /><path d="M21.532 16.5A4.5 4.5 0 0 0 17.5 10h-1.79A7.008 7.008 0 0 0 10 5.07" /></svg>
                    {{ isSaved ? '已保存' : `最近更新: ${lastSavedAgo}` }}
                  </span>
                </div>
              </div>
            </template>
          </TiptapEditor>
          <!-- 底部字数统计 -->
          <div class="note-footer">
            <span class="word-count">字数: {{ editorRef?.wordCount ?? 0 }}</span>
          </div>
        </div>
        <div v-else class="edit-panel empty-tip">
          <ElEmpty description="请选择或新增一个笔记" />
        </div>
      </ElCol>
    </ElRow>

    <!-- 右键菜单 -->
    <div v-show="contextMenu.visible" class="context-menu" :style="{ left: contextMenu.x + 'px', top: contextMenu.y + 'px' }">
      <template v-if="!contextMenu.data?.parent_id">
        <div class="context-menu-item" @click="renameNote">重命名</div>
        <div class="context-menu-item" @click="addChildFromMenu">添加文章</div>
        <div class="context-menu-item danger" @click="deleteFromMenu">删除笔记本</div>
      </template>
      <template v-else>
        <div class="context-menu-item danger" @click="deleteFromMenu">删除</div>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
  import { ElMessage, ElMessageBox } from 'element-plus'
  import { useDateFormat } from '@vueuse/core'
  import type Node from 'element-plus/es/components/tree/src/model/node'
  import request from '@/utils/http'

  defineOptions({ name: 'NotesList' })

  interface Note {
    id: number
    parent_id: number | null
    title: string
    content: string
    sort_order: number
    status: number
    update_time?: string
    children?: Note[]
  }

  const treeRef = ref()
  const editorRef = ref()
  const noteList = ref<Note[]>([])
  const currentNote = ref<Note | null>(null)
  const contextMenu = reactive({ visible: false, x: 0, y: 0, data: null as Note | null })
  const showStatus = ref(false)
  const isSaved = ref(true)
  const lastSavedTime = ref<Date>(new Date())
  const lastSavedAgo = ref('刚刚')

  function formatTime(time?: string) {
    if (!time) return ''
    return useDateFormat(time, 'YYYY-MM-DD HH:mm').value
  }

  const updateLastSavedAgo = () => {
    const diff = Math.floor((Date.now() - lastSavedTime.value.getTime()) / 1000)
    if (diff < 60) lastSavedAgo.value = '刚刚'
    else if (diff < 3600) lastSavedAgo.value = `${Math.floor(diff / 60)}分钟前`
    else lastSavedAgo.value = `${Math.floor(diff / 3600)}小时前`
  }

  const markSaved = () => {
    showStatus.value = true
    isSaved.value = true
    lastSavedTime.value = new Date()
    lastSavedAgo.value = '刚刚'
  }

  let agoTimer: ReturnType<typeof setInterval>

  const treeData = computed(() => buildTree(noteList.value))

  function buildTree(list: Note[], parentId: number | null = null): Note[] {
    return list
      .filter(item => item.parent_id === parentId)
      .map(item => ({ ...item, children: buildTree(list, item.id) }))
      .sort((a, b) => a.sort_order - b.sort_order)
  }

  async function fetchNotes() {
    const res = await request.get<Note[]>({ url: '/api/notes' })
    noteList.value = res
  }

  function onNodeClick(data: Note) {
    if (!data.parent_id && data.children?.length) {
      selectNode(data.children[0].id)
    } else if (data.parent_id) {
      currentNote.value = { ...data }
      nextTick(() => {
        showStatus.value = false
        isSaved.value = true
      })
    }
  }

  watch(
    () => currentNote.value ? `${currentNote.value.title}|${currentNote.value.content}` : '',
    () => {
      if (currentNote.value) {
        showStatus.value = true
        isSaved.value = false
        updateLastSavedAgo()
      }
    }
  )

  // 右键菜单
  function onContextMenu(event: MouseEvent, data: Note) {
    event.preventDefault()
    contextMenu.visible = true
    contextMenu.x = event.clientX
    contextMenu.y = event.clientY
    contextMenu.data = data
  }

  function hideContextMenu() {
    contextMenu.visible = false
  }

  async function renameNote() {
    hideContextMenu()
    if (!contextMenu.data) return
    const { value } = await ElMessageBox.prompt('请输入新名称', '重命名', {
      inputValue: contextMenu.data.title,
      confirmButtonText: '确定',
      cancelButtonText: '取消'
    })
    await request.put({
      url: `/api/notes/${contextMenu.data.id}`,
      data: { title: value, content: contextMenu.data.content, sort_order: contextMenu.data.sort_order, status: contextMenu.data.status }
    })
    await fetchNotes()
  }

  async function addChildFromMenu() {
    hideContextMenu()
    if (!contextMenu.data) return
    const res = await request.post<{ id: number }>({ url: '/api/notes', data: { parent_id: contextMenu.data.id, title: '新文章', status: 0 } })
    await fetchNotes()
    selectNode(res.id)
  }

  async function deleteFromMenu() {
    hideContextMenu()
    if (!contextMenu.data) return
    const msg = contextMenu.data.parent_id ? '确认删除该文章？' : '删除笔记本后所有文章也会被删除，确认？'
    await ElMessageBox.confirm(msg, '提示', { type: 'warning' })
    await request.del({ url: `/api/notes/${contextMenu.data.id}` })
    if (currentNote.value?.id === contextMenu.data.id) currentNote.value = null
    await fetchNotes()
  }

  // 拖拽：只允许二级文章拖拽
  function allowDrag(node: Node) {
    return !!node.data.parent_id
  }

  function allowDrop(draggingNode: Node, dropNode: Node, type: string) {
    // 只允许在同一笔记本内的文章之间排序
    if (type === 'inner') return false
    return !!dropNode.data.parent_id && dropNode.data.parent_id === draggingNode.data.parent_id
  }

  async function onNodeDrop(draggingNode: Node) {
    const parentId = draggingNode.data.parent_id
    const parentTreeNode = treeRef.value?.getNode(parentId)
    if (!parentTreeNode) return
    const items = parentTreeNode.childNodes.map((n: Node, i: number) => ({ id: n.data.id, sort_order: i }))
    await request.put({ url: '/api/notes/sort', data: { items } })
    await fetchNotes()
  }

  async function addRootNote() {
    const res = await request.post<{ id: number }>({ url: '/api/notes', data: { title: '新笔记本' } })
    await fetchNotes()
    selectNode(res.id)
  }

  async function saveNote(status?: number) {
    if (!currentNote.value) return
    const { id, title, content, sort_order } = currentNote.value
    const s = status ?? currentNote.value.status
    await request.put({ url: `/api/notes/${id}`, data: { title, content, sort_order, status: s } })
    currentNote.value.status = s
    ElMessage.success(s ? '发布成功' : '已保存草稿')
    markSaved()
    await fetchNotes()
  }

  function selectNode(id: number) {
    nextTick(() => {
      treeRef.value?.setCurrentKey(id)
      const node = noteList.value.find(n => n.id === id)
      if (node) {
        currentNote.value = { ...node }
        nextTick(() => {
          showStatus.value = false
          isSaved.value = true
        })
      }
    })
  }

  function onKeydown(e: KeyboardEvent) {
    if ((e.metaKey || e.ctrlKey) && e.key === 's') {
      e.preventDefault()
      saveNote()
    }
  }

  onMounted(() => {
    fetchNotes()
    document.addEventListener('click', hideContextMenu)
    document.addEventListener('keydown', onKeydown)
    agoTimer = setInterval(updateLastSavedAgo, 30000)
  })

  onBeforeUnmount(() => {
    document.removeEventListener('click', hideContextMenu)
    document.removeEventListener('keydown', onKeydown)
    clearInterval(agoTimer)
  })
</script>

<style scoped lang="scss">
  :deep(.el-col) {
    height: 100%;
  }

  .tree-panel {
    height: 100%;
    border: 1px solid var(--el-border-color);
    border-radius: 8px;
    padding: 12px;
    overflow: auto;
  }

  .tree-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .tree-title {
    font-weight: 600;
    font-size: 15px;
  }

  .tree-node-label {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    display: block;
  }

  .edit-panel {
    height: 100%;
    border: 1px solid var(--el-border-color);
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    font-family: 'DM Sans', sans-serif;

    :deep(.tiptap-editor) {
      min-height: 0;
    }

    :deep(.tiptap-content) {
      min-height: 0;
    }

    :deep(.tiptap-content .tiptap) {
      min-height: 0 !important;
    }
  }

  .note-footer {
    display: flex;
    justify-content: flex-end;
    padding: 6px 16px;
    border-top: 1px solid var(--art-gray-200);
  }

  .word-count {
    font-size: 12px;
    color: var(--art-gray-500);
  }

  .empty-tip {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .context-menu {
    position: fixed;
    z-index: 9999;
    background: var(--el-bg-color-overlay);
    border: 1px solid var(--el-border-color-light);
    border-radius: 6px;
    padding: 4px 0;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    min-width: 120px;
  }

  .context-menu-item {
    padding: 6px 16px;
    font-size: 13px;
    cursor: pointer;
    white-space: nowrap;

    &:hover {
      background: var(--el-fill-color-light);
    }

    &.danger {
      color: var(--el-color-danger);
    }
  }
</style>

<style lang="scss">
  .edit-panel {
    .title-area {
      padding: 16px 16px 0;
    }

    .title-input {
      width: 100%;
      border: none;
      outline: none;
      font-size: 24px;
      font-weight: 600;
      font-family: 'DM Sans', sans-serif;
      color: var(--art-gray-900);
      background: transparent;

      &::placeholder {
        color: var(--art-gray-400);
      }
    }

    .meta-row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 10px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--art-gray-200);
    }

    .meta-date {
      font-size: 13px;
      color: var(--art-gray-400);
    }

    .sync-status {
      display: flex;
      align-items: center;
      gap: 4px;
      font-size: 12px;

      &.saved {
        color: var(--el-color-success);
      }

      &.unsaved {
        color: var(--art-gray-400);
      }
    }

    .toolbar-save {
      margin-left: auto;

      .shortcut {
        margin-left: 4px;
        font-size: 11px;
        opacity: 0.6;
      }
    }

    .tiptap-toolbar {
      justify-content: center;
    }

    .tiptap-content .tiptap {
      padding: 12px 16px;
      font-family: 'DM Sans', sans-serif;
    }
  }
</style>
