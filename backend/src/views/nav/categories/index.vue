<!-- 导航分类管理页面 -->
<template>
  <div class="page-content !mb-5">
    <div class="flex justify-end mb-5">
      <ElButton type="primary" @click="openDialog()">新增分类</ElButton>
    </div>

    <ElTable :data="categoryList" stripe @selection-change="onSelectionChange">
      <ElTableColumn type="selection" width="45" />
      <ElTableColumn label="图标" width="70" align="center">
        <template #default="{ row }">
          <ArtSvgIcon v-if="row.icon" :icon="row.icon" class="text-xl" :style="{ color: row.color }" />
        </template>
      </ElTableColumn>
      <ElTableColumn label="名称" prop="name" width="160" />
      <ElTableColumn label="描述" prop="description" min-width="200" show-overflow-tooltip />
      <ElTableColumn label="排序" prop="sort_order" width="80" align="center" />
      <ElTableColumn label="颜色" width="80" align="center">
        <template #default="{ row }">
          <div class="size-5 rounded-full mx-auto" :style="{ background: row.color }" />
        </template>
      </ElTableColumn>
      <ElTableColumn label="状态" width="80" align="center">
        <template #default="{ row }">
          <ElSwitch v-model="row.status" active-value="1" inactive-value="2" @change="toggleStatus(row)" />
        </template>
      </ElTableColumn>
      <ElTableColumn label="操作" width="120" align="center">
        <template #default="{ row }">
          <ElButton link type="primary" :icon="EditIcon" @click="openDialog(row)" />
          <ElButton link type="danger" :icon="DeleteIcon" @click="handleDelete(row.id)" />
        </template>
      </ElTableColumn>
    </ElTable>

    <!-- 新增/编辑弹窗 -->
    <ElDialog v-model="dialogVisible" :title="editingId ? '编辑分类' : '新增分类'" width="480px">
      <ElForm :model="form" label-width="70px">
        <ElFormItem label="名称">
          <ElInput v-model="form.name" placeholder="请输入分类名称" />
        </ElFormItem>
        <ElFormItem label="图标">
          <ElInput v-model="form.icon" placeholder="如 ri:tools-line" />
        </ElFormItem>
        <ElFormItem label="颜色">
          <ElInput v-model="form.color" placeholder="#377dff" />
        </ElFormItem>
        <ElFormItem label="描述">
          <ElInput v-model="form.description" placeholder="请输入描述" />
        </ElFormItem>
        <ElFormItem label="排序">
          <ElInputNumber v-model="form.sort_order" :min="0" />
        </ElFormItem>
        <ElFormItem label="状态">
          <ElSwitch v-model="form.status" active-value="1" inactive-value="2" />
        </ElFormItem>
      </ElForm>
      <template #footer>
        <ElButton @click="dialogVisible = false">取消</ElButton>
        <ElButton type="primary" @click="submitForm">确定</ElButton>
      </template>
    </ElDialog>
  </div>
</template>

<script setup lang="ts">
  import { Edit, Delete } from '@element-plus/icons-vue'
  import { ElMessageBox } from 'element-plus'
  import request from '@/utils/http'

  defineOptions({ name: 'NavCategories' })

  const EditIcon = markRaw(Edit)
  const DeleteIcon = markRaw(Delete)

  interface NavCategory {
    id: number
    name: string
    value: string
    icon: string
    color: string
    description: string
    sort_order: number
    status: string
  }

  const categoryList = ref<NavCategory[]>([])
  const selectedIds = ref<number[]>([])
  const dialogVisible = ref(false)
  const editingId = ref<number | null>(null)

  const defaultForm = () => ({
    name: '',
    value: '',
    icon: '',
    color: '#377dff',
    description: '',
    sort_order: 0,
    status: '1'
  })
  const form = ref(defaultForm())

  const getCategories = async () => {
    const res = await request.get<NavCategory[]>({ url: '/api/nav/categories' })
    categoryList.value = res
  }

  const onSelectionChange = (rows: NavCategory[]) => {
    selectedIds.value = rows.map(r => r.id)
  }

  const openDialog = (row?: NavCategory) => {
    if (row) {
      editingId.value = row.id
      form.value = {
        name: row.name,
        value: row.value,
        icon: row.icon,
        color: row.color,
        description: row.description,
        sort_order: row.sort_order,
        status: row.status
      }
    } else {
      editingId.value = null
      form.value = defaultForm()
    }
    dialogVisible.value = true
  }

  const submitForm = async () => {
    if (!form.value.name) {
      ElMessage.error('请填写分类名称')
      return
    }
    const data = { ...form.value, value: form.value.value || form.value.name }
    if (editingId.value) {
      await request.put({ url: `/api/nav/categories/${editingId.value}`, data })
      ElMessage.success('保存成功')
    } else {
      await request.post({ url: '/api/nav/categories', data })
      ElMessage.success('添加成功')
    }
    dialogVisible.value = false
    getCategories()
  }

  const toggleStatus = async (row: NavCategory) => {
    await request.put({
      url: `/api/nav/categories/${row.id}`,
      data: { name: row.name, value: row.value, icon: row.icon, color: row.color, description: row.description, sort_order: row.sort_order, status: row.status }
    })
    ElMessage.success('状态更新成功')
  }

  const handleDelete = async (id: number) => {
    await ElMessageBox.confirm('确定删除该分类？', '提示', { type: 'warning' })
    await request.del({ url: `/api/nav/categories/${id}` })
    ElMessage.success('删除成功')
    getCategories()
  }

  onMounted(() => {
    getCategories()
  })
</script>
