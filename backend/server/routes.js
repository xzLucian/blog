import { Router } from 'express'
import pool from './db.js'
import multer from 'multer'
import path from 'path'
import { fileURLToPath } from 'url'
import fs from 'fs'
import bcrypt from 'bcryptjs'
import jwt from 'jsonwebtoken'

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const uploadDir = path.join(__dirname, 'uploads')
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true })

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, `${Date.now()}-${file.originalname}`)
})
const upload = multer({ storage })

const router = Router()

// 登录
router.post('/api/auth/login', async (req, res) => {
  const { userName, password } = req.body
  try {
    const [rows] = await pool.query('SELECT * FROM users WHERE user_name = ?', [userName])
    if (!rows.length || !bcrypt.compareSync(password, rows[0].password)) {
      return res.json({ code: 401, data: null, msg: '用户名或密码错误' })
    }
    const user = rows[0]
    const token = jwt.sign({ userId: user.id, userName: user.user_name }, JWT_SECRET, { expiresIn: '7d' })
    const refreshToken = jwt.sign({ userId: user.id }, JWT_SECRET, { expiresIn: '30d' })
    res.json({ code: 200, data: { token, refreshToken }, msg: 'ok' })
  } catch (err) {
    res.json({ code: 500, data: null, msg: '服务器错误' })
  }
})

// 用户信息
router.get('/api/user/info', async (req, res) => {
  const token = req.headers.authorization?.replace('Bearer ', '')
  try {
    const decoded = jwt.verify(token, JWT_SECRET)
    const [rows] = await pool.query('SELECT id, user_name, email, avatar FROM users WHERE id = ?', [decoded.userId])
    if (!rows.length) return res.json({ code: 401, data: null, msg: '用户不存在' })
    const user = rows[0]
    res.json({
      code: 200,
      data: {
        userId: user.id,
        userName: user.user_name,
        email: user.email,
        avatar: user.avatar
      },
      msg: 'ok'
    })
  } catch {
    res.json({ code: 401, data: null, msg: 'token无效' })
  }
})

// 菜单列表
router.get('/api/v3/system/menus/simple', (req, res) => {
  res.json({
    code: 200,
    data: [
      {
        name: 'Dashboard',
        path: '/dashboard',
        component: '/index/index',
        meta: { title: 'menus.dashboard.title', icon: 'ri:pie-chart-line' },
        children: [
          { path: 'console', name: 'Console', component: '/dashboard/console', meta: { title: 'menus.dashboard.console', keepAlive: false, fixedTab: true } }
        ]
      },
      {
        path: '/article',
        name: 'Article',
        component: '/index/index',
        meta: { title: 'menus.article.title', icon: 'ri:book-2-line' },
        children: [
          { path: 'article-list', name: 'ArticleList', component: '/article/list', meta: { title: 'menus.article.articleList', icon: 'ri:article-line', keepAlive: true } },
          { path: 'detail/:id', name: 'ArticleDetail', component: '/article/detail', meta: { title: 'menus.article.articleDetail', isHide: true, keepAlive: true, activePath: '/article/article-list' } },
          { path: 'publish', name: 'ArticlePublish', component: '/article/publish', meta: { title: 'menus.article.articlePublish', icon: 'ri:telegram-2-line', keepAlive: true } }
        ]
      },
      {
        path: '/nav',
        name: 'Nav',
        component: '/index/index',
        meta: { title: 'menus.nav.title', icon: 'ri:compass-3-line' },
        children: [
          { path: 'links', name: 'NavLinks', component: '/nav/links', meta: { title: 'menus.nav.links', icon: 'ri:link', keepAlive: true } },
          { path: 'categories', name: 'NavCategories', component: '/nav/categories', meta: { title: 'menus.nav.categories', icon: 'ri:folder-line', keepAlive: true } }
        ]
      },
      {
        path: '/notes',
        name: 'Notes',
        component: '/index/index',
        meta: { title: 'menus.notes.title', icon: 'ri:sticky-note-line' },
        children: [
          { path: 'list', name: 'NotesList', component: '/notes/list', meta: { title: 'menus.notes.list', icon: 'ri:file-list-line', keepAlive: true } }
        ]
      },
      {
        path: '/gallery',
        name: 'Gallery',
        component: '/index/index',
        meta: { title: 'menus.gallery.title', icon: 'ri:image-line' },
        children: [
          { path: 'list', name: 'GalleryList', component: '/gallery/list', meta: { title: 'menus.gallery.list', icon: 'ri:gallery-line', keepAlive: true } }
        ]
      }
    ],
    msg: 'ok'
  })
})

// 文章列表
router.get('/api/articles', async (req, res) => {
  const { page = 1, size = 40, keyword, year, status } = req.query
  const offset = (page - 1) * size
  const params = []
  let where = 'WHERE 1=1'

  if (status) {
    where += ' AND status = ?'
    params.push(status)
  }
  if (keyword) {
    where += ' AND title LIKE ?'
    params.push(`%${keyword}%`)
  }
  if (year) {
    where += ' AND YEAR(create_time) = ?'
    params.push(year)
  }

  const [[{ total }]] = await pool.query(`SELECT COUNT(*) as total FROM articles ${where}`, params)
  const [list] = await pool.query(
    `SELECT id, blog_class, title, count, create_time, type_name, status FROM articles ${where} ORDER BY create_time DESC LIMIT ? OFFSET ?`,
    [...params, Number(size), Number(offset)]
  )

  res.json({ code: 200, data: { list, total }, msg: 'ok' })
})

// 文章分类（必须在 :id 路由之前）
router.get('/api/articles/types', async (req, res) => {
  const [rows] = await pool.query(
    'SELECT DISTINCT blog_class as id, type_name as name FROM articles WHERE type_name != "" ORDER BY type_name'
  )
  res.json({ code: 200, data: rows, msg: 'ok' })
})

// 文章详情
router.get('/api/articles/:id', async (req, res) => {
  const [[row]] = await pool.query('SELECT * FROM articles WHERE id = ?', [req.params.id])
  if (!row) return res.json({ code: 404, data: null, msg: '文章不存在' })

  await pool.query('UPDATE articles SET count = count + 1 WHERE id = ?', [req.params.id])
  res.json({ code: 200, data: row, msg: 'ok' })
})

// 新增文章
router.post('/api/articles', async (req, res) => {
  const { title, blog_class, html_content, type_name, status } = req.body
  const [result] = await pool.query(
    'INSERT INTO articles (title, blog_class, html_content, type_name, status, create_time) VALUES (?, ?, ?, ?, ?, NOW())',
    [title, blog_class, html_content, type_name, status || 'draft']
  )
  res.json({ code: 200, data: { id: result.insertId }, msg: '保存成功' })
})

// 编辑文章
router.put('/api/articles/:id', async (req, res) => {
  const { title, blog_class, html_content, type_name, status } = req.body
  await pool.query(
    'UPDATE articles SET title = ?, blog_class = ?, html_content = ?, type_name = ?, status = ? WHERE id = ?',
    [title, blog_class, html_content, type_name, status, req.params.id]
  )
  res.json({ code: 200, data: null, msg: '保存成功' })
})

// 发布文章
router.put('/api/articles/:id/publish', async (req, res) => {
  await pool.query('UPDATE articles SET status = ? WHERE id = ?', ['published', req.params.id])
  res.json({ code: 200, data: null, msg: '发布成功' })
})

// ========== 导航分类 ==========

// 分类列表
router.get('/api/nav/categories', async (req, res) => {
  const [list] = await pool.query('SELECT * FROM nav_categories ORDER BY sort_order')
  res.json({ code: 200, data: list, msg: 'ok' })
})

// 新增分类
router.post('/api/nav/categories', async (req, res) => {
  const { name, value, icon, color, description, sort_order, status } = req.body
  const [result] = await pool.query(
    'INSERT INTO nav_categories (name, value, icon, color, description, sort_order, status) VALUES (?, ?, ?, ?, ?, ?, ?)',
    [name, value || name, icon || '', color || '#377dff', description || '', sort_order || 0, status || '1']
  )
  res.json({ code: 200, data: { id: result.insertId }, msg: '添加成功' })
})

// 编辑分类
router.put('/api/nav/categories/:id', async (req, res) => {
  const { name, value, icon, color, description, sort_order, status } = req.body
  await pool.query(
    'UPDATE nav_categories SET name=?, value=?, icon=?, color=?, description=?, sort_order=?, status=? WHERE id=?',
    [name, value || name, icon || '', color || '#377dff', description || '', sort_order || 0, status || '1', req.params.id]
  )
  res.json({ code: 200, data: null, msg: '保存成功' })
})

// 删除分类
router.delete('/api/nav/categories/:id', async (req, res) => {
  await pool.query('DELETE FROM nav_categories WHERE id = ?', [req.params.id])
  res.json({ code: 200, data: null, msg: '删除成功' })
})

// ========== 导航链接 ==========

// 链接列表
router.get('/api/nav/links', async (req, res) => {
  const { page = 1, size = 20, keyword, category_id } = req.query
  const offset = (page - 1) * size
  const params = []
  let where = 'WHERE 1=1'

  if (keyword) {
    where += ' AND l.title LIKE ?'
    params.push(`%${keyword}%`)
  }
  if (category_id) {
    where += ' AND l.category_id = ?'
    params.push(category_id)
  }

  const [[{ total }]] = await pool.query(`SELECT COUNT(*) as total FROM nav_links l ${where}`, params)
  const [list] = await pool.query(
    `SELECT l.*, c.name as category_name FROM nav_links l LEFT JOIN nav_categories c ON l.category_id = c.id ${where} ORDER BY l.id DESC LIMIT ? OFFSET ?`,
    [...params, Number(size), Number(offset)]
  )
  res.json({ code: 200, data: { list, total }, msg: 'ok' })
})

// 新增链接
router.post('/api/nav/links', async (req, res) => {
  const { category_id, icon, title, description, link, info, status } = req.body
  const [result] = await pool.query(
    'INSERT INTO nav_links (category_id, icon, title, description, link, info, status) VALUES (?, ?, ?, ?, ?, ?, ?)',
    [category_id, icon || '', title, description || '', link, info || '', status || '1']
  )
  res.json({ code: 200, data: { id: result.insertId }, msg: '添加成功' })
})

// 编辑链接
router.put('/api/nav/links/:id', async (req, res) => {
  const { category_id, icon, title, description, link, info, status } = req.body
  await pool.query(
    'UPDATE nav_links SET category_id=?, icon=?, title=?, description=?, link=?, info=?, status=? WHERE id=?',
    [category_id, icon || '', title, description || '', link, info || '', status || '1', req.params.id]
  )
  res.json({ code: 200, data: null, msg: '保存成功' })
})

// 删除链接
router.delete('/api/nav/links/:id', async (req, res) => {
  await pool.query('DELETE FROM nav_links WHERE id = ?', [req.params.id])
  res.json({ code: 200, data: null, msg: '删除成功' })
})

// ========== 笔记管理 ==========

// 笔记列表
router.get('/api/notes', async (req, res) => {
  const [list] = await pool.query('SELECT * FROM notes ORDER BY sort_order')
  res.json({ code: 200, data: list, msg: 'ok' })
})

// 新增笔记
router.post('/api/notes', async (req, res) => {
  const { parent_id, title, content, sort_order, status } = req.body
  const [result] = await pool.query(
    'INSERT INTO notes (parent_id, title, content, sort_order, status) VALUES (?, ?, ?, ?, ?)',
    [parent_id || null, title || '', content || '', sort_order || 0, status ?? 1]
  )
  res.json({ code: 200, data: { id: result.insertId }, msg: '添加成功' })
})

// 批量排序
router.put('/api/notes/sort', async (req, res) => {
  const { items } = req.body
  if (!Array.isArray(items)) return res.json({ code: 400, data: null, msg: '参数错误' })
  for (const { id, sort_order } of items) {
    await pool.query('UPDATE notes SET sort_order=? WHERE id=?', [sort_order, id])
  }
  res.json({ code: 200, data: null, msg: '排序成功' })
})

// 编辑笔记
router.put('/api/notes/:id', async (req, res) => {
  const { title, content, sort_order, status } = req.body
  await pool.query(
    'UPDATE notes SET title=?, content=?, sort_order=?, status=? WHERE id=?',
    [title || '', content || '', sort_order || 0, status ?? 1, req.params.id]
  )
  res.json({ code: 200, data: null, msg: '保存成功' })
})

// 删除笔记
router.delete('/api/notes/:id', async (req, res) => {
  await pool.query('DELETE FROM notes WHERE id = ?', [req.params.id])
  res.json({ code: 200, data: null, msg: '删除成功' })
})

// ========== 工作台统计 ==========
router.get('/api/dashboard/stats', async (req, res) => {
  try {
    const [[{ articles }]] = await pool.query('SELECT COUNT(*) as articles FROM articles')
    const [[{ notes }]] = await pool.query('SELECT COUNT(*) as notes FROM notes')
    const [[{ navLinks }]] = await pool.query('SELECT COUNT(*) as navLinks FROM nav_links')
    const images = fs.existsSync(uploadDir) ? fs.readdirSync(uploadDir).length : 0
    res.json({ code: 200, data: { articles, notes, navLinks, images }, msg: 'ok' })
  } catch (e) {
    console.error('dashboard stats error:', e)
    res.json({ code: 200, data: { articles: 0, notes: 0, navLinks: 0, images: 0 }, msg: 'ok' })
  }
})

// 按月统计
router.get('/api/dashboard/monthly', async (req, res) => {
  try {
    const year = new Date().getFullYear()
    const [articleRows] = await pool.query(
      'SELECT MONTH(create_time) as m, COUNT(*) as c FROM articles WHERE YEAR(create_time)=? GROUP BY m', [year]
    )
    const toArray = (rows) => {
      const arr = new Array(12).fill(0)
      rows.forEach(r => { arr[r.m - 1] = r.c })
      return arr
    }
    const [topArticles] = await pool.query(
      'SELECT title, count FROM articles ORDER BY count DESC LIMIT 8'
    )
    res.json({ code: 200, data: { articles: toArray(articleRows), topArticles }, msg: 'ok' })
  } catch (e) {
    console.error('dashboard monthly error:', e)
    res.json({ code: 200, data: { articles: new Array(12).fill(0), topArticles: [] }, msg: 'ok' })
  }
})

// ========== 图片管理 ==========

// 图片列表
router.get('/api/images', async (req, res) => {
  const { keyword } = req.query
  let where = ''
  const params = []
  if (keyword) {
    where = 'WHERE name LIKE ?'
    params.push(`%${keyword}%`)
  }
  const [list] = await pool.query(`SELECT * FROM images ${where} ORDER BY sort_order`, params)
  res.json({ code: 200, data: list, msg: 'ok' })
})

// 上传图片
router.post('/api/images', upload.single('file'), async (req, res) => {
  if (!req.file) return res.json({ code: 400, data: null, msg: '请选择文件' })
  const url = `/uploads/${req.file.filename}`
  const name = req.file.originalname
  const [[{ maxSort }]] = await pool.query('SELECT COALESCE(MAX(sort_order),0) as maxSort FROM images')
  const [result] = await pool.query(
    'INSERT INTO images (url, name, sort_order) VALUES (?, ?, ?)',
    [url, name, maxSort + 1]
  )
  res.json({ code: 200, data: { id: result.insertId, url, name, sort_order: maxSort + 1 }, msg: '上传成功' })
})

// 批量排序
router.put('/api/images/sort', async (req, res) => {
  const { items } = req.body
  if (!Array.isArray(items)) return res.json({ code: 400, data: null, msg: '参数错误' })
  for (const { id, sort_order } of items) {
    await pool.query('UPDATE images SET sort_order=? WHERE id=?', [sort_order, id])
  }
  res.json({ code: 200, data: null, msg: '排序成功' })
})

// 删除图片
router.delete('/api/images/:id', async (req, res) => {
  const [[row]] = await pool.query('SELECT url FROM images WHERE id=?', [req.params.id])
  if (row) {
    const filePath = path.join(__dirname, row.url)
    if (fs.existsSync(filePath)) fs.unlinkSync(filePath)
  }
  await pool.query('DELETE FROM images WHERE id=?', [req.params.id])
  res.json({ code: 200, data: null, msg: '删除成功' })
})

// ========== 图片上传 ==========
router.post('/api/upload', upload.single('file'), (req, res) => {
  if (!req.file) return res.json({ code: 400, data: null, msg: '请选择文件' })
  const url = `/uploads/${req.file.filename}`
  res.json({ code: 200, data: { url }, msg: '上传成功' })
})

export default router
