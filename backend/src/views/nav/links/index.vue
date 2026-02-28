<!-- 导航管理页面 -->
<template>
  <div class="page-content !mb-5">
    <ElRow justify="space-between" :gutter="10">
      <ElCol :lg="6" :md="6" :sm="14" :xs="16">
        <ElInput
          v-model="searchVal"
          :prefix-icon="Search"
          clearable
          placeholder="输入导航名称查询"
          @keyup.enter="searchLinks"
        />
      </ElCol>
      <ElCol :lg="6" :md="6" :sm="10" :xs="8" style="display: flex; justify-content: end">
        <ElButton type="primary" @click="openDialog()">新增导航</ElButton>
      </ElCol>
    </ElRow>

    <ElTable :data="linkList" class="mt-5" stripe @selection-change="onSelectionChange">
      <ElTableColumn type="selection" width="45" />
      <ElTableColumn label="图标" width="70" align="center">
        <template #default="{ row }">
          <img
            v-if="row.icon && !row._iconError"
            :src="row.icon"
            class="size-6 rounded object-contain mx-auto block"
            @error="row._iconError = true"
          />
          <div v-else class="size-6 rounded bg-gray-100 flex items-center justify-center mx-auto">
            <ArtSvgIcon icon="ri:global-line" class="text-sm text-gray-400" />
          </div>
        </template>
      </ElTableColumn>
      <ElTableColumn label="名称" prop="title" min-width="140" />
      <ElTableColumn label="描述" prop="description" min-width="200" show-overflow-tooltip />
      <ElTableColumn label="链接" prop="link" min-width="200" show-overflow-tooltip />
      <ElTableColumn label="分类" prop="category_name" width="120" />
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

    <div class="flex justify-center mt-5">
      <ElPagination
        background
        v-model:current-page="currentPage"
        v-model:page-size="pageSize"
        :page-sizes="[10, 20, 50, 100]"
        layout="total, prev, pager, next, sizes"
        :total="total"
        @current-change="getLinks"
        @size-change="getLinks"
      />
    </div>

    <!-- 新增/编辑弹窗 -->
    <ElDialog v-model="dialogVisible" :title="editingId ? '编辑导航' : '新增导航'" width="520px">
      <ElForm :model="form" label-width="70px">
        <ElFormItem label="名称">
          <ElInput v-model="form.title" placeholder="请输入导航名称" />
        </ElFormItem>
        <ElFormItem label="链接">
          <ElInput v-model="form.link" placeholder="请输入链接地址" />
        </ElFormItem>
        <ElFormItem label="图标">
          <ElInput v-model="form.icon" placeholder="请输入图标URL" />
        </ElFormItem>
        <ElFormItem label="描述">
          <ElInput v-model="form.description" placeholder="请输入描述" />
        </ElFormItem>
        <ElFormItem label="分类">
          <ElSelect v-model="form.category_id" placeholder="请选择分类" style="width: 100%">
            <ElOption
              v-for="c in categories"
              :key="c.id"
              :label="c.name"
              :value="c.id"
            />
          </ElSelect>
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
  import { Search, Edit, Delete } from '@element-plus/icons-vue'
  import { ElMessageBox } from 'element-plus'
  import request from '@/utils/http'

  defineOptions({ name: 'NavLinks' })

  const EditIcon = markRaw(Edit)
  const DeleteIcon = markRaw(Delete)

  interface NavLink {
    id: number
    category_id: number | null
    icon: string
    title: string
    description: string
    link: string
    info: string
    status: string
    category_name: string
  }

  interface NavCategory {
    id: number
    name: string
  }

  const searchVal = ref('')
  const linkList = ref<NavLink[]>([])
  const categories = ref<NavCategory[]>([])
  const currentPage = ref(1)
  const pageSize = ref(10)
  const total = ref(0)
  const selectedIds = ref<number[]>([])
  const dialogVisible = ref(false)
  const editingId = ref<number | null>(null)

  const defaultForm = () => ({
    title: '',
    link: '',
    icon: '',
    description: '',
    category_id: null as number | null,
    info: '',
    status: '1'
  })
  const form = ref(defaultForm())

  const getLinks = async () => {
    const res = await request.get<{ list: NavLink[]; total: number }>({
      url: '/api/nav/links',
      params: {
        page: currentPage.value,
        size: pageSize.value,
        keyword: searchVal.value || undefined
      }
    })
    linkList.value = res.list
    total.value = res.total
  }

  const getCategories = async () => {
    const res = await request.get<NavCategory[]>({ url: '/api/nav/categories' })
    categories.value = res
  }

  const onSelectionChange = (rows: NavLink[]) => {
    selectedIds.value = rows.map(r => r.id)
  }

  const searchLinks = () => {
    currentPage.value = 1
    getLinks()
  }

  const openDialog = (row?: NavLink) => {
    if (row) {
      editingId.value = row.id
      form.value = {
        title: row.title,
        link: row.link,
        icon: row.icon,
        description: row.description,
        category_id: row.category_id,
        info: row.info,
        status: row.status
      }
    } else {
      editingId.value = null
      form.value = defaultForm()
    }
    dialogVisible.value = true
  }

  const submitForm = async () => {
    if (!form.value.title || !form.value.link) {
      ElMessage.error('请填写名称和链接')
      return
    }
    if (editingId.value) {
      await request.put({ url: `/api/nav/links/${editingId.value}`, data: form.value })
      ElMessage.success('保存成功')
    } else {
      await request.post({ url: '/api/nav/links', data: form.value })
      ElMessage.success('添加成功')
    }
    dialogVisible.value = false
    getLinks()
  }

  const toggleStatus = async (row: NavLink) => {
    await request.put({ url: `/api/nav/links/${row.id}`, data: { category_id: row.category_id, icon: row.icon, title: row.title, description: row.description, link: row.link, info: row.info, status: row.status } })
    ElMessage.success('状态更新成功')
  }

  const handleDelete = async (id: number) => {
    await ElMessageBox.confirm('确定删除该导航？', '提示', { type: 'warning' })
    await request.del({ url: `/api/nav/links/${id}` })
    ElMessage.success('删除成功')
    getLinks()
  }

  onMounted(() => {
    getLinks()
    getCategories()
  })
</script>
