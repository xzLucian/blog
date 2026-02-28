# ByteSize

一个全栈个人博客系统，包含前台展示和后台管理两部分。

## 技术栈

**前台 (frontend)**
- Nuxt 3 + TypeScript
- Pinia 状态管理
- VueUse
- SCSS
- Shiki 代码高亮

**后台管理 (backend)**
- Vue 3 + TypeScript + Vite
- Element Plus UI 组件库
- Tailwind CSS
- Tiptap 富文本编辑器
- ECharts 数据可视化
- Vue I18n 国际化
- Vue Router + Pinia

**服务端 (backend/server)**
- Express.js
- MySQL (mysql2)
- JWT 认证
- Multer 文件上传
- bcryptjs 密码加密

## 功能模块

- 文章 (Posts) — 博客文章的发布与展示
- 笔记 (Notes) — 短内容记录
- 相册 (Photos) — 图片展示
- 友链 (Links) — 友情链接管理

## 快速开始

```bash
# 前台
cd frontend
npm install
npm run dev        # http://localhost:3100

# 后台管理
cd backend
npm install
npm run dev

# 服务端
cd backend/server
npm install
npm run dev        # http://localhost:3000
```

服务端需要配置 `.env` 文件连接 MySQL 数据库。
